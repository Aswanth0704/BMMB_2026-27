/*

  EXACT (scanned) dura mater, ONE needle-puncture wound.

  Copied from dura_cyl_two_wound_staged.cpp - the production two-wound cylinder
  driver, which is unchanged - and re-based on a real, non-cylindrical geometry
  read from Gmsh. Physics, parameters, the homeostasis derivations, the severity
  ramp, the dt ladder, checkpointing and restart are verbatim. Only the geometry
  is different, and every cylinder-specific site is replaced by the pre-processing
  in include/exact_geometry.h:

    inner/outer/cap surfaces     two edge-connected components of the lateral
                                 boundary faces + the shipped end-node lists
    fibre frame a0/s0/n0         two Laplace solves (through-thickness, axial)
    prestretch target            affine axial 1.098 + in-plane homothety 1.035,
                                 through-thickness 0.880 about a smoothed
                                 closest-point mid-surface, tapered to affine at
                                 the caps (see the header for the measured spread)
    wound                        the labelled physical volume (tag 101 "wound1"),
                                 at full severity, with a tanh skirt outside it
    needle line / free patch     through the wound centroid along its mean n0

  The scan is treated as the UNLOADED geometry (user decision, 2026-09-29), so the
  two-phase workflow is kept: phase 1 settles the target prestretch on every
  surface node, phase 2 seeds the wound and heals with the patch released.

  The mesh is aligned in place (bottom cap centroid -> origin, axis -> +z), so
  the outputs sit in the cylinder conventions: z in [0, L], wound near mid-length.

  OUTPUT NAMING IS A CONTRACT: the healing leg is prefixed wound_1_ (what
  the post-processing globs); settle_ and w1ramp<k>_ must not
  begin with wound_. wound_centers.txt carries the centre NODE ID plus the needle
  line, in the aligned frame.

  Runtime overrides (in addition to the inherited model/solver knobs):
    WOUND_MESH        .msh file                    (default dura_only_double_wound_4t.msh)
    WOUND_TOPNODES    top end-node list            (default top_end_nodes_<suffix>.txt)
    WOUND_BOTNODES    bottom end-node list         (default bottom_end_nodes_<suffix>.txt)
                      <suffix> = mesh name after "dura_only_", e.g. double_wound_4t
    WOUND_TAG         physical tag of the wound    (default 101)
    WOUND_TFINAL      healing duration [h]         (default 504 = 3 weeks)
    WOUND_TSETTLE     phase-1 duration [h]         (default 100)
    WOUND_PATCH       free-patch radius / r_w      (default 4; r_w measured off the tag)
    WOUND_WSMOOTH     skirt width [mm]             (default 1.7 x local mean edge)
    WOUND_RW          needle-track radius [mm]. Replaces the labelled region by the
                      cylinder driver's puncture: severity 0.5*(1 - tanh((d - r_w)/w))
                      about the needle line through the tag centroid, on the punctured
                      wall only, with the free patch limited to that wall as well.
                      Use WOUND_RW=0.25 WOUND_WSMOOTH=0.085 to match campaign_v2.
                      (default: off, the labelled region is the wound)
    WOUND_MIDSMOOTH   smoothing passes on the mid-surface correction (default 10)
    WOUND_TAPER       correction taper distance from the caps [mm]  (default 2)
    WOUND_INNERFREE   phase 1 pins outer + caps only; the inner surface finds its
                      own thickness (fallback if the settle gate fails)
    WOUND_NOALIGN     keep scanner coordinates
    WOUND_OUT         output prefix                (default exact)
    WOUND_NOWOUND, WOUND_CHECKONLY, WOUND_RESTART, WOUND_NOPRESTRETCH, WOUND_FREEZEX,
    WOUND_SEVRAMP/SEVSTEPS, WOUND_DTRAMP/RUNGS, WOUND_TOL/TOL_INC/MAXITER, WOUND_TRHO,
    WOUND_ALPHA_*, WOUND_TAULAMP, WOUND_LOCALSUB, WOUND_BANDW: as in the staged driver.
*/

#include <omp.h>
#include "file_io.h"
#include "file_io_gmsh.h"
#include "exact_geometry.h"
#include "wound.h"
#include "solver.h"
#include "myMeshGenerator.h"
#include "element_functions.h"
#include "local_solver.h"
#include "mechanosensing.h"
#include "homeostasis.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <ctime>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
using namespace Eigen;

//---------------------------------------------------------------------------//
// Diagnostic: recompute theta_e and H at every integration point directly from
// the tissue state, and report the spread of the nodal fields. Deliberately
// independent of the solver internals so it acts as a check on them.
//---------------------------------------------------------------------------//
static void reportState(const tissue& myTissue, const std::vector<Vector4d>& IP,
                        const char* label)
{
    const int elem_size = (int)myTissue.vol_elem_connectivity[0].size();
    const int IP_size   = (int)IP.size();

    double th_min= 1e30, th_max=-1e30, th_sum=0.0;
    double H_min = 1e30, H_max =-1e30, H_sum =0.0;
    double Je_min= 1e30, Je_max=-1e30;
    long   n=0;
    std::vector<double> th_all; th_all.reserve((size_t)myTissue.n_vol_elem*IP_size);

    const double vartheta_e  = myTissue.global_parameters[16];
    const double gamma_theta = myTissue.global_parameters[17];

    for(int ei=0; ei<myTissue.n_vol_elem; ei++){
        const std::vector<int>& e = myTissue.vol_elem_connectivity[ei];
        for(int ip=0; ip<IP_size; ip++){
            const double xi=IP[ip](0), eta=IP[ip](1), zeta=IP[ip](2);
            std::vector<double> Rxi, Reta, Rzeta;
            if(elem_size==4){
                Rxi  = evalShapeFunctionsTetRxi(xi,eta,zeta);
                Reta = evalShapeFunctionsTetReta(xi,eta,zeta);
                Rzeta= evalShapeFunctionsTetRzeta(xi,eta,zeta);
            } else {
                throw std::runtime_error("reportState: this driver supports linear tets only");
            }
            Vector3d dxdxi=Vector3d::Zero(), dxdeta=Vector3d::Zero(), dxdzeta=Vector3d::Zero();
            for(int ni=0; ni<elem_size; ni++){
                dxdxi   += myTissue.node_x[e[ni]]*Rxi[ni];
                dxdeta  += myTissue.node_x[e[ni]]*Reta[ni];
                dxdzeta += myTissue.node_x[e[ni]]*Rzeta[ni];
            }
            Matrix3d dxdXi;
            dxdXi << dxdxi(0), dxdeta(0), dxdzeta(0),
                     dxdxi(1), dxdeta(1), dxdzeta(1),
                     dxdxi(2), dxdeta(2), dxdzeta(2);
            const Matrix3d FF = dxdXi*myTissue.elem_jac_IP[ei][ip].transpose();
            const Matrix3d CC = FF.transpose()*FF;
            const double J = FF.determinant();

            const int g = ei*IP_size+ip;
            const Vector3d& n0     = myTissue.ip_n0[g];
            const Vector3d& lamdaP = myTissue.ip_lamdaP[g];

            const double th = evalThetaE(J, CC.inverse(), n0, lamdaP(0), lamdaP(1));
            const double H  = evalHe(th, vartheta_e, gamma_theta);
            const double Je = J/(lamdaP(0)*lamdaP(1)*lamdaP(2));

            th_min=std::min(th_min,th); th_max=std::max(th_max,th); th_sum+=th;
            H_min =std::min(H_min ,H ); H_max =std::max(H_max ,H ); H_sum +=H;
            Je_min=std::min(Je_min,Je); Je_max=std::max(Je_max,Je);
            th_all.push_back(th);
            n++;
        }
    }

    auto span = [](const std::vector<double>& v, double& lo, double& hi, double& mean){
        lo=1e30; hi=-1e30; mean=0.0;
        for(double x : v){ lo=std::min(lo,x); hi=std::max(hi,x); mean+=x; }
        if(!v.empty()) mean/=(double)v.size();
    };
    double rlo,rhi,rmean, clo,chi,cmean, plo,phi_hi,pmean, alo,ahi,amean;
    span(myTissue.node_rho,   rlo,rhi,rmean);
    span(myTissue.node_c,     clo,chi,cmean);
    span(myTissue.ip_phif,    plo,phi_hi,pmean);
    span(myTissue.node_alpha, alo,ahi,amean);

    std::cout<<"\n================ STATE REPORT: "<<label<<" ================\n";
    std::cout<<std::fixed<<std::setprecision(6);
    std::cout<<"  theta_e  min/mean/max : "<<th_min<<" / "<<th_sum/n<<" / "<<th_max
             <<"   (p1 "<<percentile(th_all,1.0)<<", p99 "<<percentile(th_all,99.0)<<")\n";
    std::cout<<"  H        min/mean/max : "<<H_min <<" / "<<H_sum /n<<" / "<<H_max <<"\n";
    std::cout<<"  det(F^e) min/max      : "<<Je_min<<" / "<<Je_max<<"   (1 if incompressible)\n";
    std::cout<<"  rho      min/mean/max : "<<rlo<<" / "<<rmean<<" / "<<rhi<<"\n";
    std::cout<<"  c        min/mean/max : "<<clo<<" / "<<cmean<<" / "<<chi<<"\n";
    std::cout<<"  phi      min/mean/max : "<<plo<<" / "<<pmean<<" / "<<phi_hi<<"\n";
    std::cout<<"  alpha    min/mean/max : "<<alo<<" / "<<amean<<" / "<<ahi<<"\n";
    if(rlo<0.0 || clo<0.0 || plo<0.0 || alo<0.0)
        std::cout<<"  *** WARNING: negative concentration - solver has no clamping ***\n";
    std::cout<<"===========================================================\n\n";
    std::cout.unsetf(std::ios::fixed);
}

int main(int argc, char *argv[])
{
    Eigen::initParallel();
    std::cout<<"\nRunning full domain simulations with " << Eigen::nbThreads( ) << " threads.\n";
    srand (time(NULL));

    const bool verbose    = (std::getenv("WOUND_VERBOSE")   != nullptr);
    const bool check_only = (std::getenv("WOUND_CHECKONLY") != nullptr);
    auto env_str = [](const char* key, const std::string& fallback){
        const char* v = std::getenv(key);
        return (v && *v) ? std::string(v) : fallback;
    };
    auto env_dbl = [](const char* key, double fallback){
        const char* v = std::getenv(key);
        return (v && *v) ? std::atof(v) : fallback;
    };

    //=======================================================================//
    // NORMALIZED HOMEOSTATIC STATE
    //=======================================================================//
    // Everything is normalized by its physiological value, so homeostasis is
    // (alpha, rho, c, phi) = (0, 1, 1, 1).
    const double rho_h = 1.0;
    const double c_h   = 1.0;
    const double phi_h = 1.0;

    // values for the wound (normalized, homeostasis = 1)
    double rho_wound   = 1.0e-4;
    double c_wound     = 1.0e-4;
    double phif0_wound = 1.0e-2;
    // Injury releases the pro-inflammatory signal, so alpha starts HIGH in the
    // wound and decays (half-life ln2/d_alpha ~ 54 h), driving cytokine
    // production through p_c_alpha.
    double alpha_wound = 1.0;
    double kappa0_wound = 1./3;      // uniform dispersion

    // values for the healthy tissue
    double rho_healthy   = rho_h;
    double c_healthy     = c_h;
    double alpha_healthy = 0.0;
    double phif0_healthy = phi_h;
    double kappa0_healthy = 0.024;   // fiber dispersion, from our own experiments

    //=======================================================================//
    // MECHANOSENSING
    //=======================================================================//
    // theta_e is the in-plane AREAL elastic stretch ||cof(F^e).n0||.
    // vartheta_e is the measured adult dural areal prestretch 1.098*1.035, so
    // H(vartheta_e) = 1/2 exactly in the healthy prestretched state. Every
    // derived parameter below assumes H_h = 1/2.
    double vartheta_e  = 1.136;
    double gamma_theta = 10.0;
    const double H_h   = 0.5;

    //=======================================================================//
    // GLOBAL PARAMETERS  (verbatim from dura_cyl_two_wound_staged.cpp)
    //=======================================================================//
    double k0 = 0.02;      // neo-Hookean ground substance [MPa]
    double kf = 40.0;      // collagen fiber stiffness [MPa]
    double k2 = 0.048;     // fiber exponential coefficient [-]

    double t_rho   = env_dbl("WOUND_TRHO", 1.28571e-3);  // [MPa] at rho = rho_h
    double t_rho_c = t_rho*3.28571;                      // enhancement by cytokine
    std::cout<<"active traction: t_rho="<<t_rho<<" t_rho_c="<<t_rho_c<<" MPa\n";
    double K_t     = 0.2;                     // saturation of traction by collagen
    double K_t_c   = c_h/10.0;                // saturation of traction by cytokine

    double D_rhorho = 0.0;      // UNUSED: global_parameters[7] is never read.
    double D_rhoc   = 0.0;      // fibroblast chemotaxis - neglected in this model
    double D_cc     = 0.00930;  // cytokine diffusion [mm^2/h]

    double p_rho       = 0.0154;            // baseline proliferation
    double p_rho_c     = 1.48*p_rho;        // cytokine-driven (ratio 1.48)
    double p_rho_theta = 0.109*p_rho;       // mechano-driven  (ratio 0.109)
    double K_rho_c     = 1.31;              // saturation of proliferation by c
    double d_rho       = 0.00369;           // apoptosis

    double d_c     = 0.00386;               // decay
    double K_c_c   = 1.20;                  // saturation of production by c
    double r_c_e   = 0.447;                 // p_c_e / p_c_rho (fixed ratio)

    double bx = 0.0, by = 0.0, bz = 0.0;    // body force

    double D_alpha   = env_dbl("WOUND_ALPHA_D",     0.00930); // [mm^2/h] = D_c
    double d_alpha   = env_dbl("WOUND_ALPHA_DECAY", 0.0128);  // [1/h]
    double p_c_alpha = env_dbl("WOUND_ALPHA_PC",    0.208);   // [1/h]

    //=======================================================================//
    // LOCAL PARAMETERS
    //=======================================================================//
    double p_phi       = 9.34e-4;           // baseline collagen production [1/h]
    double p_phi_c     = 1.41e-3;           // cytokine-driven
    double p_phi_theta = 4.96*p_phi;        // mechano-driven (ratio 4.96)
    double K_phi_c     = 1.08;              // saturation of deposition by c
    double d_phi       = 2.02e-3;           // baseline degradation [1/h]
    double d_phi_rho_c = 2.87e-4;           // degradation coupled to c and rho

    //=======================================================================//
    // DERIVED PARAMETERS - enforce (0,1,1,1) as an exact fixed point
    //=======================================================================//
    double K_rho_rho = deriveKrhorho(p_rho,p_rho_c,p_rho_theta,K_rho_c,d_rho,
                                     rho_h,c_h,H_h);
    double K_phi_rho = deriveKphirho(p_phi,p_phi_c,p_phi_theta,K_phi_c,
                                     d_phi,d_phi_rho_c,rho_h,c_h,phi_h,H_h);
    double p_c_rho    = derivePcrho(d_c,K_c_c,r_c_e,H_h);
    double p_c_thetaE = r_c_e*p_c_rho;

    if(!(K_phi_rho > 0.0))
        throw std::runtime_error("K_phi_rho <= 0: collagen production too weak "
                                 "relative to degradation for a physiological fixed point.");
    if(!(K_rho_rho > rho_h))
        throw std::runtime_error("K_rho_rho <= rho_h: apoptosis exceeds the "
                                 "homeostatic proliferation budget.");

    std::vector<double> global_parameters = {k0,kf,k2,t_rho,t_rho_c,K_t,K_t_c,
        D_rhorho,D_rhoc,D_cc,p_rho,p_rho_c,p_rho_theta,K_rho_c,K_rho_rho,d_rho,
        vartheta_e,gamma_theta,p_c_rho,p_c_thetaE,K_c_c,d_c,bx,by,bz,
        D_alpha,d_alpha,p_c_alpha};

    double tau_omega = 10./(K_phi_rho+1);
    double tau_kappa = 1./(K_phi_rho+1);
    double gamma_kappa = 5.;
    const double tau_lamdaP_scale = env_dbl("WOUND_TAULAMP", 1.0);
    double tau_lamdaP_a = 0.05*tau_lamdaP_scale;
    double tau_lamdaP_s = 0.05*tau_lamdaP_scale;
    double tau_lamdaP_n = 0.05*tau_lamdaP_scale;

    double tol_local = 1e-8;        // inert in the explicit local solver
    double time_step_ratio = env_dbl("WOUND_LOCALSUB", 25);
    double max_iter = 100;          // inert in the explicit local solver

    // Deadband for plastic growth: must contain (1.098, 1.035, 0.880).
    double lamdaE_lo = 0.85;
    double lamdaE_hi = 1.15;
    double lamdaE_bandw = env_dbl("WOUND_BANDW", 0.002);
    {
        const double lam_r_h = 1.0/(1.098*1.035);
        if(lamdaE_lo > lam_r_h || lamdaE_hi < 1.098)
            throw std::runtime_error("plastic-growth deadband excludes the "
                "physiological prestretch: healthy tissue would remodel.");
    }

    std::vector<double> local_parameters = {p_phi,p_phi_c,p_phi_theta,K_phi_c,
        K_phi_rho,d_phi,d_phi_rho_c,tau_omega,tau_kappa,gamma_kappa,
        tau_lamdaP_a,tau_lamdaP_s,tau_lamdaP_n,vartheta_e,gamma_theta,
        tol_local,time_step_ratio,max_iter,lamdaE_lo,lamdaE_hi,lamdaE_bandw};

    //---------------------------------//
    std::cout<<"\n===================== PARAMETERS (normalized) =====================\n";
    std::cout<<std::scientific<<std::setprecision(6);
    const char* gnames[] = {"k0","kf","k2","t_rho","t_rho_c","K_t","K_t_c",
        "D_rhorho(unused)","D_rhoc","D_cc","p_rho","p_rho_c","p_rho_theta",
        "K_rho_c","K_rho_rho*","d_rho","vartheta_e","gamma_theta","p_c_rho*",
        "p_c_thetaE*","K_c_c","d_c","bx","by","bz",
        "D_alpha","d_alpha","p_c_alpha"};
    for(size_t i=0;i<global_parameters.size();i++)
        std::cout<<"  global["<<std::setw(2)<<i<<"] "<<std::setw(18)<<gnames[i]
                 <<" = "<<global_parameters[i]<<"\n";
    const char* lnames[] = {"p_phi","p_phi_c","p_phi_theta","K_phi_c","K_phi_rho*",
        "d_phi","d_phi_rho_c","tau_omega*","tau_kappa*","gamma_kappa",
        "tau_lamdaP_a","tau_lamdaP_s","tau_lamdaP_n","vartheta_e","gamma_theta",
        "tol_local","time_step_ratio","max_iter","lamdaE_lo","lamdaE_hi","lamdaE_bandw"};
    for(size_t i=0;i<local_parameters.size();i++)
        std::cout<<"  local ["<<std::setw(2)<<i<<"] "<<std::setw(18)<<lnames[i]
                 <<" = "<<local_parameters[i]<<"\n";
    std::cout<<"  (* = derived to enforce homeostasis)\n";

    {
        double S_rho   = evalSrho(p_rho,p_rho_c,p_rho_theta,K_rho_c,K_rho_rho,d_rho,
                                  rho_h,c_h,H_h);
        double S_c     = evalSc(p_c_rho,p_c_thetaE,K_c_c,d_c,p_c_alpha,
                                rho_h,c_h,alpha_healthy,H_h);
        double S_alpha = evalSalpha(d_alpha,alpha_healthy);
        double S_phi   = evalPhidot(p_phi,p_phi_c,p_phi_theta,K_phi_c,K_phi_rho,
                                    d_phi,d_phi_rho_c,rho_h,c_h,phi_h,H_h);
        std::cout<<"\n  Fixed-point residuals at (alpha,rho,c,phi)=(0,1,1,1), H=1/2:\n";
        std::cout<<"    s_rho   = "<<S_rho<<"\n    s_c     = "<<S_c
                 <<"\n    s_alpha = "<<S_alpha
                 <<"\n    s_phi   = "<<S_phi<<"\n";
        const double worst = std::max(std::max(std::fabs(S_rho),std::fabs(S_c)),
                                      std::max(std::fabs(S_alpha),std::fabs(S_phi)));
        if(worst > 1e-12)
            std::cout<<"  *** WARNING: homeostasis not exact (worst "<<worst<<") ***\n";
        else
            std::cout<<"  homeostasis exact to machine precision.\n";
    }
    std::cout<<"===================================================================\n\n";
    std::cout.unsetf(std::ios::scientific);

    //=======================================================================//
    // GEOMETRY AND MESH  (exact dura from Gmsh)
    //=======================================================================//
    std::string mesh_filename = env_str("WOUND_MESH", "dura_only_double_wound_4t.msh");
    std::string list_suffix;
    {
        // "dura_only_double_wound_4t.msh" -> "double_wound_4t"
        std::string base = mesh_filename;
        const size_t slash = base.find_last_of('/');
        if(slash != std::string::npos) base = base.substr(slash+1);
        const std::string key = "dura_only_";
        const size_t p = base.find(key);
        const size_t d = base.rfind(".msh");
        if(p != std::string::npos && d != std::string::npos && d > p+key.size())
            list_suffix = "_" + base.substr(p+key.size(), d-(p+key.size()));
    }
    const std::string top_list = env_str("WOUND_TOPNODES", "top_end_nodes"    + list_suffix + ".txt");
    const std::string bot_list = env_str("WOUND_BOTNODES", "bottom_end_nodes" + list_suffix + ".txt");
    const int wound_tag = (int)env_dbl("WOUND_TAG", 101);
    std::cout<<"mesh file: "<<mesh_filename<<"\n  end-node lists: "<<bot_list<<" (bottom), "
             <<top_list<<" (top)\n  wound = physical tag "<<wound_tag<<"\n";

    HexMesh myMesh = readGmshMsh22doubleWound(mesh_filename, bot_list, top_list, 101, 104, 102);

    ExactGeomParams gp;
    gp.smooth_passes = (int)env_dbl("WOUND_MIDSMOOTH", 10);
    gp.taper         = env_dbl("WOUND_TAPER", 2.0);
    gp.align         = (std::getenv("WOUND_NOALIGN") == nullptr);
    const ExactGeometry geom = buildExactGeometry(myMesh, bot_list, top_list, gp);

    std::cout<<"Mesh: "<<myMesh.n_nodes<<" nodes, "<<myMesh.n_elements<<" elements, "
             <<"z in [0, "<<geom.length<<"] after alignment\n";
    if(verbose){
        std::cout<<"nodes\n";
        for(int nodei=0;nodei<myMesh.n_nodes;nodei++)
            std::cout<<myMesh.nodes[nodei](0)<<","<<myMesh.nodes[nodei](1)<<","
                     <<myMesh.nodes[nodei](2)<<"\n";
    }

    int elem_size = myMesh.elements[0].size();
    if(elem_size != 4)
        throw std::runtime_error("this driver supports linear tets only (tet10 quadrature is broken)");
    std::vector<Vector4d> IP = LineQuadriIPTet();
    int IP_size = IP.size();

    //=======================================================================//
    // PRESTRETCH (target from the geometry module)
    //=======================================================================//
    const double lam_z  = gp.lam_z;
    const double lam_th = gp.lam_th;
    std::cout<<"prestretch: lam_axial="<<lam_z<<" lam_circ="<<lam_th
             <<" lam_thick="<<1.0/(lam_z*lam_th)
             <<"  -> theta_e="<<lam_z*lam_th<<"\n";
    const bool no_prestretch = (std::getenv("WOUND_NOPRESTRETCH") != nullptr);
    const bool inner_free    = (std::getenv("WOUND_INNERFREE")   != nullptr);

    //=======================================================================//
    // WOUND: the labelled region
    //=======================================================================//
    // Membership comes from the physical tag, decided ONCE on the reference mesh.
    // The needle line (centroid + mean normal of the tagged elements) is used for
    // the free patch and for reporting; r_w is the tagged region's in-plane radius.
    std::vector<char> elem_is_wound(myMesh.n_elements, 0);
    std::vector<char> node_is_wound(myMesh.n_nodes, 0);
    std::vector<int>  wound_elems, wound_nodes;
    Vector3d c_w = Vector3d::Zero(), n_w = Vector3d::Zero();
    double r_wound = 0.0;             // in-plane radius of the tagged region
    double mean_edge_global = 0.0, mean_edge_local = 0.0;
    {
        if(myMesh.element_domain_id.size() != (size_t)myMesh.n_elements)
            throw std::runtime_error("mesh has no element_domain_id - the wound tag cannot be read");
        double vol_w = 0.0;
        long ng = 0, nl = 0;
        for(int e=0;e<myMesh.n_elements;e++){
            const std::vector<int>& el = myMesh.elements[e];
            const bool w = (myMesh.element_domain_id[e] == wound_tag);
            for(size_t a=0;a<el.size();a++)
                for(size_t b=a+1;b<el.size();b++){
                    const double L = (myMesh.nodes[el[a]] - myMesh.nodes[el[b]]).norm();
                    mean_edge_global += L; ng++;
                    if(w){ mean_edge_local += L; nl++; }
                }
            if(!w) continue;
            elem_is_wound[e] = 1; wound_elems.push_back(e);
            for(int a=0;a<4;a++) node_is_wound[el[a]] = 1;
            Matrix3d J; for(int j=0;j<3;j++) J.col(j) = myMesh.nodes[el[j+1]] - myMesh.nodes[el[0]];
            const double V = std::fabs(J.determinant())/6.0;
            vol_w += V;
            n_w += V*geom.elem_n0[e];
        }
        if(ng) mean_edge_global /= (double)ng;
        mean_edge_local = nl ? mean_edge_local/(double)nl : mean_edge_global;
        if(wound_elems.empty())
            throw std::runtime_error("no elements carry the wound tag - check WOUND_TAG against the mesh");
        n_w.normalize();
        for(int i=0;i<myMesh.n_nodes;i++) if(node_is_wound[i]){ wound_nodes.push_back(i); c_w += myMesh.nodes[i]; }
        c_w /= (double)wound_nodes.size();
        for(int i : wound_nodes){
            const Vector3d d = myMesh.nodes[i] - c_w;
            r_wound = std::max(r_wound, (d - d.dot(n_w)*n_w).norm());
        }
    }
    // distance from the needle line, in the REFERENCE (aligned) frame
    auto d_line = [&](const Vector3d& X)->double{
        const Vector3d d = X - c_w;
        return (d - d.dot(n_w)*n_w).norm();
    };

    // WOUND_RW: the cylinder driver's needle track instead of the labelled region.
    // The site (tag centroid) and the needle direction are kept, so the puncture
    // stays where it was; only its radius and its severity law change, to those of
    // dura_cyl_two_wound_staged.cpp. The labelled region is about 0.35 x 0.25 mm at
    // full severity, so a run without this option is NOT the 25-gauge puncture of
    // the cylinder results and cannot be compared with them quantitatively.
    const double r_tag = r_wound;
    const double rw_override = env_dbl("WOUND_RW", -1.0);
    const bool analytic_track = (rw_override > 0.0);
    if(analytic_track) r_wound = rw_override;
    // The needle line also crosses the opposite wall of the sac. Anything further
    // than this from the centroid ALONG the needle is not the punctured wall (the
    // wall is at most 0.55 mm thick; the opposite wall is about 10 mm away).
    const double wall_reach = 2.0;
    auto on_punctured_wall = [&](const Vector3d& X)->bool{
        return std::fabs((X - c_w).dot(n_w)) <= wall_reach;
    };

    // Smoothing width of the skirt outside the labelled region.
    const double w_override = env_dbl("WOUND_WSMOOTH", -1.0);
    const double w_smooth = (w_override > 0.0) ? w_override : 1.7*mean_edge_local;
    std::cout<<"element edge: global mean "<<mean_edge_global<<" mm, wound-region mean "
             <<mean_edge_local<<" mm; skirt width w_smooth "<<w_smooth<<" mm = "
             <<w_smooth/mean_edge_local<<" local elements\n";
    if(w_smooth < 1.2*mean_edge_local)
        std::cout<<"  *** WARNING: skirt width under-resolved against the LOCAL element size;"
                   " expect Galerkin undershoot at the wound edge ***\n";

    // Severity: 1 inside the labelled region, 1 - tanh(d/w) outside, where d is
    // the distance to the nearest labelled node. The labelled region IS the wound;
    // the skirt only smooths its edge so the discretization does not see a step.
    auto nearest_wound_node_dist = [&](const Vector3d& X)->double{
        double best = 1e30;
        for(int i : wound_nodes) best = std::min(best, (X - myMesh.nodes[i]).squaredNorm());
        return std::sqrt(best);
    };
    auto skirt = [&](double d)->double{ return 1.0 - std::tanh(d/w_smooth); };
    // WOUND_RW: severity of the needle track, the law of the cylinder driver
    // (severity_ref_k), evaluated on the reference mesh about the needle line.
    auto track = [&](const Vector3d& X)->double{
        if(!on_punctured_wall(X)) return 0.0;
        return 0.5*(1.0 - std::tanh((d_line(X) - r_wound)/w_smooth));
    };

    std::vector<double> wound_sev_node(myMesh.n_nodes, 0.0);
    std::vector<double> wound_sev_ip(myMesh.n_elements*IP_size, 0.0);
    int centre_node = -1;
    {
        double best = 1e30;
        int n_core_nodes = 0, n_inner = 0, n_outer = 0, n_skirt = 0;
        int n_track_half = 0, n_track_any = 0, n_track_ip_half = 0;   // WOUND_RW counts
        for(int i=0;i<myMesh.n_nodes;i++){
            double sev;
            if(node_is_wound[i]){
                sev = 1.0; n_core_nodes++;
                if(geom.surface_code[i] == 1) n_inner++;
                if(geom.surface_code[i] == 2) n_outer++;
                const double dc = (myMesh.nodes[i] - c_w).squaredNorm();
                if(dc < best){ best = dc; centre_node = i; }
            } else {
                // cheap reject: nothing beyond r_w + 4 w_smooth from the line matters
                const double dl = d_line(myMesh.nodes[i]);
                if(dl > r_wound + 4.0*w_smooth + 0.5) sev = 0.0;
                else sev = skirt(nearest_wound_node_dist(myMesh.nodes[i]));
                if(sev > 1e-3) n_skirt++;
            }
            if(analytic_track){
                // the tag above still picks the centre node and reports the site;
                // the severity itself comes from the needle track
                sev = track(myMesh.nodes[i]);
                if(sev > 0.5)  n_track_half++;
                if(sev > 1e-3) n_track_any++;
            }
            wound_sev_node[i] = sev;
        }
        int n_ip_core = 0, n_ip_skirt = 0;
        for(int e=0;e<myMesh.n_elements;e++){
            for(int ip=0;ip<IP_size;ip++){
                const int g = e*IP_size+ip;
                if(analytic_track){
                    std::vector<double> Rt = evalShapeFunctionsTetR(IP[ip](0),IP[ip](1),IP[ip](2));
                    Vector3d X_t = Vector3d::Zero();
                    for(int a=0;a<4;a++) X_t += Rt[a]*myMesh.nodes[myMesh.elements[e][a]];
                    const double sev_t = track(X_t);
                    wound_sev_ip[g] = sev_t;
                    if(sev_t > 0.5) n_track_ip_half++;
                    continue;
                }
                if(elem_is_wound[e]){ wound_sev_ip[g] = 1.0; n_ip_core++; continue; }
                std::vector<double> R = evalShapeFunctionsTetR(IP[ip](0),IP[ip](1),IP[ip](2));
                Vector3d X_IP = Vector3d::Zero();
                for(int a=0;a<4;a++) X_IP += R[a]*myMesh.nodes[myMesh.elements[e][a]];
                const double dl = d_line(X_IP);
                double sev = 0.0;
                if(dl <= r_wound + 4.0*w_smooth + 0.5) sev = skirt(nearest_wound_node_dist(X_IP));
                wound_sev_ip[g] = sev;
                if(sev > 1e-3) n_ip_skirt++;
            }
        }
        std::cout<<"\nwound (tag "<<wound_tag<<") on the REFERENCE mesh: "<<wound_elems.size()
                 <<" elements, "<<n_core_nodes<<" nodes at full severity, "<<n_ip_core
                 <<" integration points; skirt reaches "<<n_skirt<<" more nodes / "
                 <<n_ip_skirt<<" IPs (sev > 1e-3)\n"
                 <<"  centroid        : ("<<c_w(0)<<", "<<c_w(1)<<", "<<c_w(2)<<")  z/L = "
                 <<c_w(2)/geom.length<<"\n"
                 <<"  needle direction: ("<<n_w(0)<<", "<<n_w(1)<<", "<<n_w(2)<<")  (mean n0 of the region)\n"
                 <<"  in-plane radius : r_w = "<<r_wound<<" mm\n"
                 <<"  centre node     : "<<centre_node<<" at ("<<myMesh.nodes[centre_node](0)<<", "
                 <<myMesh.nodes[centre_node](1)<<", "<<myMesh.nodes[centre_node](2)<<"), "
                 <<std::sqrt(best)<<" mm from the centroid\n"
                 <<"  wall coverage   : "<<n_inner<<" inner-surface and "<<n_outer
                 <<" outer-surface nodes in the region\n";
        if(n_inner == 0 || n_outer == 0)
            std::cout<<"  *** WARNING: PARTIAL-THICKNESS lesion - the labelled region does not"
                       " reach both surfaces. Fix the MESH, not the seeding logic. ***\n";
        if(d_line(myMesh.nodes[centre_node]) > 0.5*r_wound)
            std::cout<<"  note: centre node sits "<<d_line(myMesh.nodes[centre_node])<<" mm off the needle line\n";
        if(analytic_track)
            std::cout<<"  *** WOUND_RW: the labelled region (in-plane radius "<<r_tag<<" mm) only fixes the\n"
                     <<"      site and the needle direction. The wound is the needle track r_w = "<<r_wound
                     <<" mm,\n      severity 0.5*(1 - tanh((d - r_w)/"<<w_smooth<<")), punctured wall only: "
                     <<n_track_half<<" nodes and "<<n_track_ip_half<<" integration points at severity > 0.5, "
                     <<n_track_any<<" nodes at severity > 1e-3 ***\n";
    }

    // Surface nodes. In phase 1 every surface node is pinned to the target (the
    // cylinder pins end rings + both lateral surfaces); WOUND_INNERFREE leaves the
    // inner surface free so the thickness comes from the constitutive response.
    std::vector<char> on_boundary(myMesh.n_nodes, 0), pin_phase1(myMesh.n_nodes, 0);
    {
        int n_cap=0, n_in=0, n_out=0, n_pin=0;
        for(int i=0;i<myMesh.n_nodes;i++){
            const int code = geom.surface_code[i];
            on_boundary[i] = (code != 0) ? 1 : 0;
            if(code==3) n_cap++; else if(code==1) n_in++; else if(code==2) n_out++;
            pin_phase1[i] = (code != 0 && !(inner_free && code == 1)) ? 1 : 0;
            if(pin_phase1[i]) n_pin++;
        }
        std::cout<<"boundary nodes: "<<n_cap<<" on the caps, "<<n_in<<" inner, "<<n_out
                 <<" outer; phase 1 pins "<<n_pin<<(inner_free ? " (WOUND_INNERFREE: inner surface free)":"")<<"\n";
    }

    std::vector<Vector3d> node_target(myMesh.n_nodes);
    for(int nodei=0;nodei<myMesh.n_nodes;nodei++)
        node_target[nodei] = no_prestretch ? myMesh.nodes[nodei] : geom.target[nodei];
    if(no_prestretch){
        std::cout<<"\n*** WOUND_NOPRESTRETCH: prestretch DISABLED (target = reference) ***\n"
                 <<"    theta_e = 1 so H = 0.204, not the 0.5 that K_rho_rho, K_phi_rho and\n"
                 <<"    p_c_rho are all derived from. (0,1,1,1) is therefore NOT a fixed\n"
                 <<"    point and the far field WILL drift. Convergence test only.\n\n";
    }

    //=======================================================================//
    // INITIAL CONDITIONS - healthy everywhere (the wound comes in phase 2)
    //=======================================================================//
    std::vector<double> node_rho0(myMesh.n_nodes, rho_healthy);
    std::vector<double> node_c0  (myMesh.n_nodes, c_healthy);
    std::vector<double> node_alpha0(myMesh.n_nodes, alpha_healthy);
    std::vector<double> ip_phi0  (myMesh.n_elements*IP_size, phif0_healthy);
    std::vector<double> ip_kappa0(myMesh.n_elements*IP_size, kappa0_healthy);
    Vector3d lamda0_healthy(1.,1.,1.);
    std::vector<Vector3d> ip_a00(myMesh.n_elements*IP_size);
    std::vector<Vector3d> ip_s00(myMesh.n_elements*IP_size);
    std::vector<Vector3d> ip_n00(myMesh.n_elements*IP_size);
    std::vector<Vector3d> ip_lamda0(myMesh.n_elements*IP_size, lamda0_healthy);

    // Fibre frame from the Laplace solves: one gradient per linear tet, so the
    // frame is the same at its four integration points. Kept in ip_*00 as the
    // reference frame the wound reseed falls back to.
    for(int elemi=0;elemi<myMesh.n_elements;elemi++){
        for(int ip=0;ip<IP_size;ip++){
            ip_a00[elemi*IP_size+ip] = geom.elem_a0[elemi];
            ip_s00[elemi*IP_size+ip] = geom.elem_s0[elemi];
            ip_n00[elemi*IP_size+ip] = geom.elem_n0[elemi];
        }
    }

    //=======================================================================//
    // BOUNDARY CONDITIONS - phase 1: prestretch held on the surface
    //=======================================================================//
    std::map<int,double> eBC_x, eBC_rho, eBC_c, eBC_alpha;
    for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
        if(!on_boundary[nodei]) continue;
        if(pin_phase1[nodei]){
            eBC_x.insert(std::pair<int,double>(nodei*3+0, node_target[nodei](0)));
            eBC_x.insert(std::pair<int,double>(nodei*3+1, node_target[nodei](1)));
            eBC_x.insert(std::pair<int,double>(nodei*3+2, node_target[nodei](2)));
        }
        // species held at their healthy values on the caps only; the lateral
        // surfaces are free surfaces for transport
        if(myMesh.boundary_flag[nodei] == 1){
            eBC_rho.insert(std::pair<int,double>(nodei, rho_healthy));
            eBC_c.insert  (std::pair<int,double>(nodei, c_healthy));
            eBC_alpha.insert(std::pair<int,double>(nodei, alpha_healthy));
        }
    }
    std::map<int,double> nBC_x, nBC_rho, nBC_c, nBC_alpha; // unused: never read by the solver

    //=======================================================================//
    // ASSEMBLE THE TISSUE
    //=======================================================================//
    tissue myTissue;
    myTissue.vol_elem_connectivity  = myMesh.elements;
    myTissue.surf_elem_connectivity = myMesh.surface_elements;
    myTissue.global_parameters = global_parameters;
    myTissue.local_parameters  = local_parameters;
    myTissue.boundary_flag = myMesh.boundary_flag;
    myTissue.surface_boundary_flag = myMesh.surface_boundary_flag;
    myTissue.node_X = myMesh.nodes;          // UNLOADED (aligned scan) reference - never re-referenced
    myTissue.node_x = node_target;           // start at the prestretched guess
    myTissue.node_rho_0 = node_rho0;  myTissue.node_rho = node_rho0;
    myTissue.node_c_0   = node_c0;    myTissue.node_c   = node_c0;
    myTissue.node_alpha_0 = node_alpha0; myTissue.node_alpha = node_alpha0;
    myTissue.ip_phif_0  = ip_phi0;    myTissue.ip_phif  = ip_phi0;
    myTissue.ip_a0_0    = ip_a00;     myTissue.ip_a0    = ip_a00;
    myTissue.ip_s0_0    = ip_s00;     myTissue.ip_s0    = ip_s00;
    myTissue.ip_n0_0    = ip_n00;     myTissue.ip_n0    = ip_n00;
    myTissue.ip_kappa_0 = ip_kappa0;  myTissue.ip_kappa = ip_kappa0;
    myTissue.ip_lamdaP_0= ip_lamda0;  myTissue.ip_lamdaP= ip_lamda0;
    myTissue.ip_lamdaE  = ip_lamda0;
    myTissue.ip_strain = std::vector<Matrix3d>(myMesh.n_elements*IP_size, Matrix3d::Identity());
    myTissue.ip_stress = std::vector<Matrix3d>(myMesh.n_elements*IP_size, Matrix3d::Zero());
    myTissue.eBC_x = eBC_x;  myTissue.eBC_rho = eBC_rho;  myTissue.eBC_c = eBC_c;
    myTissue.eBC_alpha = eBC_alpha;
    myTissue.nBC_x = nBC_x;  myTissue.nBC_rho = nBC_rho;  myTissue.nBC_c = nBC_c;
    myTissue.nBC_alpha = nBC_alpha;
    myTissue.time       = 0.0;
    myTissue.time_step  = 0.2;
    myTissue.tol        = env_dbl("WOUND_TOL", 1e-6);
    myTissue.tol_inc    = env_dbl("WOUND_TOL_INC", 1e-4);
    myTissue.max_iter   = (int)env_dbl("WOUND_MAXITER", 200);
    myTissue.n_node     = myMesh.n_nodes;
    myTissue.n_vol_elem = myMesh.n_elements;
    myTissue.n_surf_elem= myMesh.n_surf_elements;
    myTissue.n_IP       = IP_size*myMesh.n_elements;

    std::cout<<"filling dofs...\n";
    fillDOFmap(myTissue);
    std::cout<<"going to eval jacobians...\n";
    evalElemJacobians(myTissue);
    std::cout<<"element jacobians: "<<myTissue.elem_jac_IP.size()<<"\n";
    std::cout<<"Total :"<<myTissue.n_dof<<" dof\n";

    std::vector<int> save_node; save_node.clear();
    std::vector<int> save_ip;   save_ip.clear();
    // The HEALING leg is named wound_1_ unconditionally (analysis-script contract).
    std::string out_prefix = env_str("WOUND_OUT", "exact");
    const std::string heal_prefix = "wound_1_";

    // Wound-centre node id, needle line and radius, in the ALIGNED reference frame.
    {
        std::ofstream wc("wound_centers.txt");
        wc<<"# wound_index node_id ref_x ref_y ref_z  centroid_x centroid_y centroid_z"
             "  needle_nx needle_ny needle_nz  r_w  (aligned frame: bottom cap centroid at origin, axis +z)\n";
        wc<<std::setprecision(10)<<1<<" "<<centre_node<<" "
          <<myMesh.nodes[centre_node](0)<<" "<<myMesh.nodes[centre_node](1)<<" "<<myMesh.nodes[centre_node](2)<<" "
          <<c_w(0)<<" "<<c_w(1)<<" "<<c_w(2)<<" "<<n_w(0)<<" "<<n_w(1)<<" "<<n_w(2)<<" "<<r_wound<<"\n";
        std::cout<<"wrote wound_centers.txt (aligned reference frame)\n";
    }

    // Geometry diagnostics, then the BC / IC check, before solving anything.
    writeExactGeometryVTK(myMesh, geom, out_prefix + "_FRAME.vtk", out_prefix + "_TARGET.vtk");
    {
        std::string f1 = out_prefix + "_BCCHECK.vtk";
        std::string f2 = out_prefix + "_second_BCCHECK.vtk";
        writeParaview(myTissue, f1.c_str(), f2.c_str());
        std::cout<<"wrote BC check: "<<f1<<"\n";
    }
    reportState(myTissue, IP, "initial guess (prestretch applied, before solve)");

    //=======================================================================//
    // PHASE 1: SETTLE THE PRESTRETCH (no wound)
    //=======================================================================//
    double t_settle = env_dbl("WOUND_TSETTLE", 100.0);
    const bool is_restart = (std::getenv("WOUND_RESTART") != nullptr
                             && std::string(std::getenv("WOUND_RESTART")).size() > 0);
    if(is_restart){
        std::cout<<"\n#### PHASE 1 SKIPPED (restart) ####\n";
    }
    else if(check_only){
        std::cout<<"\n#### PHASE 1 SKIPPED (WOUND_CHECKONLY: geometry check, no solve) ####\n";
    }
    else if(t_settle > 0.0){
        myTissue.time = 0.0;
        myTissue.time_final = t_settle;
        std::string f = out_prefix + "_settle_";
        std::cout<<"\n#### PHASE 1: prestretch settling for "<<t_settle
                 <<" h (no wound) ####\n";
        sparseWoundSolver(myTissue, f, 5, save_node, save_ip);
        reportState(myTissue, IP, "after prestretch settling (HOMEOSTASIS GATE)");
    }

    if(std::getenv("WOUND_NOWOUND") != nullptr){
        std::cout<<"WOUND_NOWOUND set - stopping after the homeostasis gate.\n";
        return 0;
    }

    //=======================================================================//
    // PHASE 2: SEED THE WOUND AND HEAL
    //=======================================================================//
    const double patch_mult   = env_dbl("WOUND_PATCH", 4.0);
    const double patch_radius = patch_mult*r_wound;
    const double dt_normal    = myTissue.time_step;
    const double t_final      = env_dbl("WOUND_TFINAL", 3*7*24.0);   // healing [h]

    // Rebuild the Dirichlet set with the patch around the puncture released. Held
    // surface nodes are pinned where they ARE (node_x): for nodes pinned in phase 1
    // that is the target, for a free inner surface it is the settled position, and
    // on restart it is what the checkpoint carries - so the rebuild is exact.
    auto rebuild_bcs = [&](){
        std::map<int,double> eBC_x2, eBC_rho2, eBC_c2, eBC_alpha2;
        int n_freed = 0;

        const bool freeze_x = (std::getenv("WOUND_FREEZEX") != nullptr);
        if(freeze_x){
            for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
                const Vector3d& xf = myTissue.node_x[nodei];
                eBC_x2.insert(std::pair<int,double>(nodei*3+0, xf(0)));
                eBC_x2.insert(std::pair<int,double>(nodei*3+1, xf(1)));
                eBC_x2.insert(std::pair<int,double>(nodei*3+2, xf(2)));
            }
            std::cout<<"\n*** WOUND_FREEZEX: all "<<myMesh.n_nodes<<" nodal displacements pinned"
                     <<" at the settled configuration ***\n";
        }

        for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
            if(!on_boundary[nodei]) continue;
            // Patch membership on the REFERENCE mesh, concentric with the wound by
            // construction (same needle line as the tagging report).
            // With WOUND_RW the patch is limited to the punctured wall, as the
            // cylinder driver limits it to X(0) < 0. Without it the old behaviour is
            // kept (the opposite wall inside the patch radius is released too).
            const bool near_wound = (d_line(myMesh.nodes[nodei]) < patch_radius)
                && (!analytic_track || on_punctured_wall(myMesh.nodes[nodei]));
            if(near_wound){ n_freed++; continue; }
            const Vector3d& xh = myTissue.node_x[nodei];
            eBC_x2.insert(std::pair<int,double>(nodei*3+0, xh(0)));
            eBC_x2.insert(std::pair<int,double>(nodei*3+1, xh(1)));
            eBC_x2.insert(std::pair<int,double>(nodei*3+2, xh(2)));
            if(myMesh.boundary_flag[nodei] == 1){
                eBC_rho2.insert(std::pair<int,double>(nodei, rho_healthy));
                eBC_c2.insert  (std::pair<int,double>(nodei, c_healthy));
                eBC_alpha2.insert(std::pair<int,double>(nodei, alpha_healthy));
            }
        }
        std::cout<<"\nfree patch radius "<<patch_radius<<" mm ("<<patch_mult
                 <<"x r_w) around the wound: freed "<<n_freed
                 <<" surface nodes so it can contract\n";
        myTissue.eBC_x = eBC_x2;
        myTissue.eBC_rho = eBC_rho2;
        myTissue.eBC_c = eBC_c2;
        myTissue.eBC_alpha = eBC_alpha2;
        fillDOFmap(myTissue);
        std::cout<<"Total :"<<myTissue.n_dof<<" dof after releasing the patch\n";
    };

    auto blend = [](double healthy, double wound, double sev){
        return healthy + sev*(wound - healthy);
    };

    // Baseline snapshotted at the moment of injury (keeps WOUND_SEVRAMP idempotent).
    std::vector<double> base_rho, base_c, base_alpha, base_phif, base_kappa;
    bool have_base = false;

    auto seed_wound = [&](double frac, bool announce){
        if(!have_base){
            base_rho   = myTissue.node_rho;    base_c     = myTissue.node_c;
            base_alpha = myTissue.node_alpha;  base_phif  = myTissue.ip_phif;
            base_kappa = myTissue.ip_kappa;    have_base  = true;
        }
        int nn = 0, nip = 0;
        for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
            const double sev = wound_sev_node[nodei];
            if(sev > 1e-3){
                const double r = blend(base_rho[nodei],   rho_wound,   sev);
                const double c = blend(base_c[nodei],     c_wound,     sev);
                const double a = blend(base_alpha[nodei], alpha_wound, sev);
                myTissue.node_rho_0[nodei]   = r;  myTissue.node_rho[nodei]   = r;
                myTissue.node_c_0[nodei]     = c;  myTissue.node_c[nodei]     = c;
                myTissue.node_alpha_0[nodei] = a;  myTissue.node_alpha[nodei] = a;
                if(sev > 0.5) nn++;
            }
        }
        for(int elemi=0;elemi<myMesh.n_elements;elemi++){
            for(int ip=0;ip<IP_size;ip++){
                const int g = elemi*IP_size+ip;
                const double sev = frac*wound_sev_ip[g];
                if(sev > 1e-3){
                    // The wound's fibre frame is reset to the reference (Laplace)
                    // frame - the cylinder rebuilt its analytic frame here. lamdaP
                    // is 1 in healthy tissue so the frame carries no prestretch.
                    const double ph = blend(base_phif[g],  phif0_wound,  sev);
                    const double kp = blend(base_kappa[g], kappa0_wound, sev);
                    myTissue.ip_phif_0[g] = ph;  myTissue.ip_phif[g] = ph;
                    myTissue.ip_a0_0[g]   = ip_a00[g];  myTissue.ip_a0[g]   = ip_a00[g];
                    myTissue.ip_s0_0[g]   = ip_s00[g];  myTissue.ip_s0[g]   = ip_s00[g];
                    myTissue.ip_n0_0[g]   = ip_n00[g];  myTissue.ip_n0[g]   = ip_n00[g];
                    myTissue.ip_kappa_0[g]= kp;  myTissue.ip_kappa[g]= kp;
                    if(sev > 0.5) nip++;
                }
            }
        }
        if(announce){
            std::cout<<"\nseeded wound: "<<nn<<" nodes, "<<nip
                     <<" integration points at severity > 0.5  (tag "<<wound_tag
                     <<", r_w "<<r_wound<<" mm, w_smooth "<<w_smooth<<" mm)\n";
            if(nn == 0 || nip == 0)
                std::cout<<"  *** WARNING: wound region is empty ***\n";
        }
    };

    const int    sev_rungs = (int)env_dbl("WOUND_SEVRAMP", 0);
    const int    n_rung    = (int)env_dbl("WOUND_RUNGS",   40);  // steps per ladder rung
    const double dt_start  = env_dbl("WOUND_DTRAMP", 0.002);
    const int    sev_steps = (int)env_dbl("WOUND_SEVSTEPS", 10);

    auto report_centre = [&](const char* when){
        const int nd = centre_node;
        double ph = 0.0; int cnt = 0;
        for(int e=0;e<myMesh.n_elements;e++)
            for(int a=0;a<elem_size;a++)
                if(myMesh.elements[e][a] == nd){
                    for(int ip=0;ip<IP_size;ip++){ ph += myTissue.ip_phif[e*IP_size+ip]; cnt++; }
                    break;
                }
        std::cout<<"\n---- wound centre ("<<when<<") ----\n"
                 <<"  node "<<nd<<": rho="<<myTissue.node_rho[nd]<<"  c="<<myTissue.node_c[nd]
                 <<"  alpha="<<myTissue.node_alpha[nd]<<"  phi="<<(cnt? ph/cnt : 0.0)<<"\n"
                 <<"----------------------------------\n";
    };

    // Inflict the wound, absorb the puncture transient, heal for t_target.
    auto inflict_and_heal = [&](double t_target){
        rebuild_bcs();
        seed_wound(sev_rungs > 0 ? 1.0/sev_rungs : 1.0, true);
        reportState(myTissue, IP, "wound seeded, before healing");
        report_centre("wound just inflicted");
        {
            std::string c1 = out_prefix + "_WOUND1CHECK.vtk";
            std::string c2 = out_prefix + "_second_WOUND1CHECK.vtk";
            writeParaview(myTissue, c1.c_str(), c2.c_str());
        }

        double t_ramp_total = 0.0;
        if(sev_rungs > 0 && dt_start > 0.0){
            std::cout<<"\n#### severity continuation, "<<sev_rungs<<" rungs ####\n";
            for(int r=2;r<=sev_rungs;r++){
                const double frac = (double)r/(double)sev_rungs;
                seed_wound(frac, false);
                myTissue.time       = 0.0;
                myTissue.time_step  = dt_start;
                myTissue.time_final = sev_steps*dt_start;
                std::ostringstream fs;
                fs << "w1rampsev" << r << "_";
                sparseWoundSolver(myTissue, fs.str(), std::max(1, sev_steps),
                                  save_node, save_ip);
                t_ramp_total += sev_steps*dt_start;
            }
            reportState(myTissue, IP, "after severity continuation");
        }

        if(dt_start > 0.0 && dt_start < dt_normal && n_rung > 0){
            std::cout<<"\n#### puncture transient, geometric dt ladder ####\n";
            int rung = 0;
            for(double dt = dt_start; dt < dt_normal; dt *= 5.0, ++rung){
                const double dt_use = std::min(dt, dt_normal);
                myTissue.time = 0.0;
                myTissue.time_step  = dt_use;
                myTissue.time_final = n_rung*dt_use;
                std::ostringstream fr;
                fr << "w1ramp" << rung << "_";
                std::cout<<"  rung "<<rung<<": "<<n_rung<<" steps at dt = "<<dt_use
                         <<"  ("<<n_rung*dt_use<<" h)\n";
                sparseWoundSolver(myTissue, fr.str(), std::max(1, n_rung/2),
                                  save_node, save_ip);
                t_ramp_total += n_rung*dt_use;
            }
            myTissue.time_step = dt_normal;
            reportState(myTissue, IP, "after the puncture transient");
            std::cout<<"  transient absorbed over "<<t_ramp_total<<" h\n";
        }

        myTissue.time = 0.0;
        myTissue.time_final = std::max(0.0, t_target - t_ramp_total);
        std::cout<<"\n#### HEALING for "<<myTissue.time_final
                 <<" h at dt = "<<dt_normal<<"  (prefix "<<heal_prefix<<") ####\n";
        sparseWoundSolver(myTissue, heal_prefix, 5, save_node, save_ip);
        reportState(myTissue, IP, "after healing");
        report_centre("end of the case");
    };

    if(check_only){
        rebuild_bcs();
        seed_wound(1.0, true);
        reportState(myTissue, IP, "wound seeded (check only)");
        report_centre("check only");
        std::string f1 = out_prefix + "_WOUNDCHECK.vtk";
        std::string f2 = out_prefix + "_second_WOUNDCHECK.vtk";
        writeParaview(myTissue, f1.c_str(), f2.c_str());
        std::cout<<"\n#### WOUND_CHECKONLY: wound seeded and written to "<<f1
                 <<" for inspection, then stopping. ####\n";
        return 0;
    }

    //=======================================================================//
    // RESTART: resume a killed run from its last checkpoint
    //=======================================================================//
    const std::string restart_file = env_str("WOUND_RESTART", "");
    if(!restart_file.empty()){
        std::cout<<"\n#### RESTART from "<<restart_file<<" ####\n";
        tissue ck = readTissue(restart_file.c_str());
        if(ck.n_node != myTissue.n_node || ck.n_vol_elem != myTissue.n_vol_elem
           || ck.n_IP != myTissue.n_IP){
            std::ostringstream e;
            e<<"restart checkpoint does not match this mesh: checkpoint has "
             <<ck.n_node<<" nodes / "<<ck.n_vol_elem<<" elems / "<<ck.n_IP<<" IP, "
             <<"mesh has "<<myTissue.n_node<<" / "<<myTissue.n_vol_elem<<" / "<<myTissue.n_IP
             <<". Check WOUND_MESH.";
            throw std::runtime_error(e.str());
        }
        if(restart_file.find("wound_1_") == std::string::npos)
            throw std::runtime_error("expected 'wound_1_' in the checkpoint filename");
        int step_offset = 0;
        {
            size_t d = restart_file.find_last_of('.');
            size_t u = restart_file.find_last_of('_');
            if(u != std::string::npos && d != std::string::npos && d > u)
                step_offset = std::atoi(restart_file.substr(u+1, d-u-1).c_str());
        }
        myTissue.node_x       = ck.node_x;
        myTissue.node_rho     = ck.node_rho;     myTissue.node_rho_0   = ck.node_rho_0;
        myTissue.node_c       = ck.node_c;       myTissue.node_c_0     = ck.node_c_0;
        myTissue.node_alpha   = ck.node_alpha;   myTissue.node_alpha_0 = ck.node_alpha_0;
        myTissue.ip_phif      = ck.ip_phif;      myTissue.ip_phif_0    = ck.ip_phif_0;
        myTissue.ip_a0        = ck.ip_a0;        myTissue.ip_a0_0      = ck.ip_a0_0;
        myTissue.ip_s0        = ck.ip_s0;        myTissue.ip_s0_0      = ck.ip_s0_0;
        myTissue.ip_n0        = ck.ip_n0;        myTissue.ip_n0_0      = ck.ip_n0_0;
        myTissue.ip_kappa     = ck.ip_kappa;     myTissue.ip_kappa_0   = ck.ip_kappa_0;
        myTissue.ip_lamdaP    = ck.ip_lamdaP;    myTissue.ip_lamdaP_0  = ck.ip_lamdaP_0;

        rebuild_bcs();          // held nodes pinned at the checkpoint's node_x: exact

        const double t_target = (ck.time_final > 0.0) ? ck.time_final : t_final;
        myTissue.time       = ck.time;
        myTissue.time_step  = dt_normal;
        myTissue.time_final = t_target;

        reportState(myTissue, IP, "restored from checkpoint");
        report_centre("restored from checkpoint");
        std::cout<<"\n#### RESUMING healing at t = "<<ck.time
                 <<" h, target "<<t_target<<" h ("<<(t_target-ck.time)<<" h remaining)"
                 <<", file numbering continues from "<<step_offset<<" ####\n";
        if(t_target - ck.time > 1e-9){
            sparseWoundSolver(myTissue, heal_prefix, 5, save_node, save_ip, step_offset);
            reportState(myTissue, IP, "after healing");
            report_centre("end of the case");
        } else {
            std::cout<<"  this leg was already complete at the checkpoint - nothing to do\n";
        }
        return 0;
    }

    std::cout<<"\n=======================================================\n"
             <<"  WOUND INFLICTED, healing for "<<t_final<<" h ("<<t_final/(7*24.)<<" weeks)\n"
             <<"=======================================================\n";
    inflict_and_heal(t_final);

    return 0;
}
