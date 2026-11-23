/*
  Geometry pre-processing for a scanned dura mesh. See include/exact_geometry.h for
  the overview; this file is the implementation, in the order the pipeline runs:
    1. end lists          (load, sanitise by axial coordinate)
    2. alignment          (bottom-cap centroid -> origin, axis -> +z)
    3. boundary faces     (used once), caps vs lateral, two lateral components
    4. Laplace fields     u (through-thickness), v (axial)  -> frame, thickness
    5. prestretch target  (affine + smoothed, tapered mid-surface correction)
    6. VTK writers
  Everything is deterministic given the mesh and the two lists, which is what the
  restart path relies on.
*/

#include "exact_geometry.h"

#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

using Eigen::Vector3d;
using Eigen::Matrix3d;

namespace {

typedef Eigen::Matrix<double,4,3> Grad4x3;

//---------------------------------------------------------------------------//
// small utilities
//---------------------------------------------------------------------------//
std::vector<int> loadIntList(const std::string& file)
{
    std::ifstream f(file);
    if(!f.is_open()) throw std::runtime_error("cannot open node list: " + file);
    std::vector<int> ids; int id;
    while(f >> id) ids.push_back(id);
    if(ids.empty()) throw std::runtime_error("empty node list: " + file);
    return ids;
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n/2] : 0.5*(v[n/2-1] + v[n/2]);
}

struct FaceKey {
    int a, b, c;
    bool operator==(const FaceKey& o) const { return a==o.a && b==o.b && c==o.c; }
};
struct FaceKeyHash {
    size_t operator()(const FaceKey& k) const {
        return ((size_t)k.a*73856093u) ^ ((size_t)k.b*19349663u) ^ ((size_t)k.c*83492791u);
    }
};
FaceKey makeKey(int i, int j, int k)
{
    std::array<int,3> s = {i,j,k}; std::sort(s.begin(), s.end());
    return FaceKey{s[0], s[1], s[2]};
}

struct DSU {
    std::vector<int> p;
    explicit DSU(int n) : p(n) { for(int i=0;i<n;i++) p[i]=i; }
    int find(int x){ while(p[x]!=x){ p[x]=p[p[x]]; x=p[x]; } return x; }
    void unite(int a, int b){ a=find(a); b=find(b); if(a!=b) p[a]=b; }
};

// Boundary face with outward orientation (a,b,c right-handed about the outward
// normal) and its owning tet.
struct BFace { int a, b, c, elem; };

//---------------------------------------------------------------------------//
// linear-tet shape-function gradients: G(a,k) = dN_a/dX_k, and the volume
//---------------------------------------------------------------------------//
void tetGradient(const HexMesh& mesh, int e, Grad4x3& G, double& V)
{
    const std::vector<int>& c = mesh.elements[e];
    Matrix3d J;
    for(int j=0;j<3;j++) J.col(j) = mesh.nodes[c[j+1]] - mesh.nodes[c[0]];
    V = J.determinant()/6.0;
    const Matrix3d Jinv = J.inverse();
    Eigen::Matrix<double,4,3> dNdxi;
    dNdxi << -1,-1,-1,
              1, 0, 0,
              0, 1, 0,
              0, 0, 1;
    G = dNdxi*Jinv;
}

//---------------------------------------------------------------------------//
// Laplace problem on the tets, Dirichlet 0 on `zero_nodes`, 1 on `one_nodes`,
// natural elsewhere. Reduced SPD system solved with SimplicialLDLT.
//---------------------------------------------------------------------------//
std::vector<double> solveLaplace(const HexMesh& mesh,
                                 const std::vector<Grad4x3>& G,
                                 const std::vector<double>& V,
                                 const std::vector<int>& zero_nodes,
                                 const std::vector<int>& one_nodes)
{
    const int n = mesh.n_nodes;
    std::vector<double> u(n, 0.0);
    std::vector<char>   fixed(n, 0);
    for(int i : zero_nodes){ fixed[i]=1; u[i]=0.0; }
    for(int i : one_nodes ){ fixed[i]=1; u[i]=1.0; }

    std::vector<int> red(n, -1); int nf = 0;
    for(int i=0;i<n;i++) if(!fixed[i]) red[i] = nf++;

    std::vector<Eigen::Triplet<double>> trip; trip.reserve((size_t)mesh.n_elements*16);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(nf);
    for(int e=0;e<mesh.n_elements;e++){
        const std::vector<int>& c = mesh.elements[e];
        const Eigen::Matrix<double,4,4> Ke = std::fabs(V[e])*G[e]*G[e].transpose();
        for(int a=0;a<4;a++){
            if(fixed[c[a]]) continue;
            for(int b=0;b<4;b++){
                if(fixed[c[b]]) rhs(red[c[a]]) -= Ke(a,b)*u[c[b]];
                else            trip.emplace_back(red[c[a]], red[c[b]], Ke(a,b));
            }
        }
    }
    Eigen::SparseMatrix<double> K(nf, nf);
    K.setFromTriplets(trip.begin(), trip.end());
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K);
    if(solver.info() != Eigen::Success)
        throw std::runtime_error("exact_geometry: Laplace factorisation failed");
    const Eigen::VectorXd x = solver.solve(rhs);
    if(solver.info() != Eigen::Success)
        throw std::runtime_error("exact_geometry: Laplace solve failed");
    for(int i=0;i<n;i++) if(!fixed[i]) u[i] = x(red[i]);
    return u;
}

//---------------------------------------------------------------------------//
// closest point on a triangle (Ericson, Real-Time Collision Detection 5.1.5)
//---------------------------------------------------------------------------//
Vector3d closestPointTriangle(const Vector3d& p, const Vector3d& a,
                              const Vector3d& b, const Vector3d& c)
{
    const Vector3d ab = b - a, ac = c - a, ap = p - a;
    const double d1 = ab.dot(ap), d2 = ac.dot(ap);
    if(d1 <= 0.0 && d2 <= 0.0) return a;
    const Vector3d bp = p - b;
    const double d3 = ab.dot(bp), d4 = ac.dot(bp);
    if(d3 >= 0.0 && d4 <= d3) return b;
    const double vc = d1*d4 - d3*d2;
    if(vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0){ const double v = d1/(d1-d3); return a + v*ab; }
    const Vector3d cp = p - c;
    const double d5 = ab.dot(cp), d6 = ac.dot(cp);
    if(d6 >= 0.0 && d5 <= d6) return c;
    const double vb = d5*d2 - d1*d6;
    if(vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0){ const double w = d2/(d2-d6); return a + w*ac; }
    const double va = d3*d6 - d5*d4;
    if(va <= 0.0 && (d4-d3) >= 0.0 && (d5-d6) >= 0.0){
        const double w = (d4-d3)/((d4-d3)+(d5-d6)); return b + w*(c-b);
    }
    const double denom = 1.0/(va+vb+vc);
    const double v = vb*denom, w = vc*denom;
    return a + ab*v + ac*w;
}

//---------------------------------------------------------------------------//
// uniform grid over a triangulated surface for closest-point queries
//---------------------------------------------------------------------------//
struct SurfaceGrid {
    const std::vector<Vector3d>* X = nullptr;
    const std::vector<std::array<int,3>>* F = nullptr;
    Vector3d lo; double h = 1.0;
    std::unordered_map<long long, std::vector<int>> cells;

    static long long key(int i, int j, int k){
        return ((long long)(i+ (1<<20)) << 42) ^ ((long long)(j + (1<<20)) << 21) ^ (long long)(k + (1<<20));
    }
    void cellOf(const Vector3d& p, int& i, int& j, int& k) const {
        i = (int)std::floor((p(0)-lo(0))/h); j = (int)std::floor((p(1)-lo(1))/h); k = (int)std::floor((p(2)-lo(2))/h);
    }
    void build(const std::vector<Vector3d>& nodes, const std::vector<std::array<int,3>>& faces, double cell)
    {
        X = &nodes; F = &faces; h = cell;
        lo = Vector3d( 1e30, 1e30, 1e30);
        for(const auto& f : faces) for(int t=0;t<3;t++) lo = lo.cwiseMin(nodes[f[t]]);
        lo -= Vector3d(h,h,h);
        for(size_t fi=0; fi<faces.size(); fi++){
            Vector3d bmin( 1e30, 1e30, 1e30), bmax(-1e30,-1e30,-1e30);
            for(int t=0;t<3;t++){ bmin = bmin.cwiseMin(nodes[faces[fi][t]]); bmax = bmax.cwiseMax(nodes[faces[fi][t]]); }
            int i0,j0,k0,i1,j1,k1; cellOf(bmin,i0,j0,k0); cellOf(bmax,i1,j1,k1);
            for(int i=i0;i<=i1;i++) for(int j=j0;j<=j1;j++) for(int k=k0;k<=k1;k++)
                cells[key(i,j,k)].push_back((int)fi);
        }
    }
    // closest point on the surface to p, and the distance
    Vector3d project(const Vector3d& p, double& best) const
    {
        int ci,cj,ck; cellOf(p,ci,cj,ck);
        best = 1e30; Vector3d bp = p;
        for(int r=0; r<64; r++){
            for(int i=ci-r;i<=ci+r;i++) for(int j=cj-r;j<=cj+r;j++) for(int k=ck-r;k<=ck+r;k++){
                if(std::max(std::abs(i-ci),std::max(std::abs(j-cj),std::abs(k-ck))) != r) continue;
                auto it = cells.find(key(i,j,k));
                if(it == cells.end()) continue;
                for(int fi : it->second){
                    const std::array<int,3>& f = (*F)[fi];
                    const Vector3d q = closestPointTriangle(p, (*X)[f[0]], (*X)[f[1]], (*X)[f[2]]);
                    const double d = (q-p).norm();
                    if(d < best){ best = d; bp = q; }
                }
            }
            // every face not yet visited lies in a ring >= r+1, hence at distance >= r*h
            if(best <= r*h) break;
        }
        if(best >= 1e29) throw std::runtime_error("exact_geometry: closest-point query found no face");
        return bp;
    }
};

double smoothstep01(double q){ q = std::min(1.0, std::max(0.0, q)); return q*q*(3.0-2.0*q); }

void printStat(const char* name, const std::vector<double>& v, const std::vector<double>* w)
{
    double lo=1e30, hi=-1e30, mean=0.0, wsum=0.0;
    for(size_t i=0;i<v.size();i++){
        lo = std::min(lo, v[i]); hi = std::max(hi, v[i]);
        const double wi = w ? std::fabs((*w)[i]) : 1.0;
        mean += wi*v[i]; wsum += wi;
    }
    std::cout<<"  "<<std::setw(8)<<std::left<<name<<std::right
             <<"  min "<<std::setw(8)<<lo<<"  p1 "<<std::setw(8)<<percentile(v,1.0)
             <<"  mean "<<std::setw(8)<<mean/wsum<<"  p99 "<<std::setw(8)<<percentile(v,99.0)
             <<"  max "<<std::setw(8)<<hi<<"\n";
}

} // namespace

//---------------------------------------------------------------------------//
double percentile(std::vector<double> v, double p)
{
    if(v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = (p/100.0)*(double)(v.size()-1);
    const size_t i = (size_t)std::floor(pos);
    const double t = pos - (double)i;
    if(i+1 >= v.size()) return v.back();
    return v[i]*(1.0-t) + v[i+1]*t;
}

//---------------------------------------------------------------------------//
ExactGeometry buildExactGeometry(HexMesh& mesh,
                                 const std::string& bottom_list_file,
                                 const std::string& top_list_file,
                                 const ExactGeomParams& P)
{
    ExactGeometry g;
    const int n = mesh.n_nodes;
    const int m = mesh.n_elements;
    if(m == 0 || mesh.elements[0].size() != 4)
        throw std::runtime_error("exact_geometry: expects a linear tetrahedral mesh");

    std::cout<<"\n===================== EXACT GEOMETRY =====================\n";

    //-------------------------------------------------------------------//
    // 1. end lists, sanitised by axial coordinate
    //-------------------------------------------------------------------//
    std::vector<int> top_raw = loadIntList(top_list_file);
    std::vector<int> bot_raw = loadIntList(bottom_list_file);
    for(int id : top_raw) if(id < 0 || id >= n) throw std::runtime_error("top list id out of range");
    for(int id : bot_raw) if(id < 0 || id >= n) throw std::runtime_error("bottom list id out of range");

    auto centroid = [&](const std::vector<int>& ids){
        Vector3d c = Vector3d::Zero(); for(int i : ids) c += mesh.nodes[i]; return Vector3d(c/(double)ids.size());
    };
    Vector3d ct = centroid(top_raw), cb = centroid(bot_raw);
    Vector3d ax = (ct - cb).normalized();

    auto sanitise = [&](const std::vector<int>& raw, std::vector<int>& kept, std::vector<int>& dropped, const char* name){
        std::vector<double> s; s.reserve(raw.size());
        for(int i : raw) s.push_back((mesh.nodes[i]-cb).dot(ax));
        const double med = median(s);
        for(size_t k=0;k<raw.size();k++)
            (std::fabs(s[k]-med) > P.list_tol ? dropped : kept).push_back(raw[k]);
        if(!dropped.empty()){
            std::cout<<"  "<<name<<" list: dropped "<<dropped.size()<<" stray id(s):";
            for(int i : dropped) std::cout<<" "<<i;
            std::cout<<"  (axial coordinate more than "<<P.list_tol<<" mm from the list median)\n";
        }
        if((double)dropped.size() > 0.01*(double)raw.size())
            throw std::runtime_error(std::string("exact_geometry: more than 1% of the ") + name +
                                     " end list is off its cap - check that the list matches this mesh");
    };
    sanitise(top_raw, g.top_nodes, g.dropped_top, "top");
    sanitise(bot_raw, g.bottom_nodes, g.dropped_bottom, "bottom");
    {
        std::unordered_set<int> a(g.top_nodes.begin(), g.top_nodes.end());
        for(int i : g.bottom_nodes) if(a.count(i))
            throw std::runtime_error("exact_geometry: a node is in both end lists");
    }
    ct = centroid(g.top_nodes); cb = centroid(g.bottom_nodes);
    ax = (ct - cb).normalized();
    g.length = (ct - cb).norm();
    std::cout<<"  end lists: "<<g.top_nodes.size()<<" top, "<<g.bottom_nodes.size()
             <<" bottom; cap-to-cap length "<<g.length<<" mm\n";

    //-------------------------------------------------------------------//
    // 2. alignment: bottom centroid -> origin, axis -> +z
    //-------------------------------------------------------------------//
    g.origin = cb;
    g.rot = P.align ? Eigen::Quaterniond::FromTwoVectors(ax, Vector3d::UnitZ()).toRotationMatrix()
                    : Matrix3d::Identity();
    if(P.align){
        for(int i=0;i<n;i++) mesh.nodes[i] = g.rot*(mesh.nodes[i] - g.origin);
        std::cout<<"  aligned: origin "<<g.origin.transpose()<<", axis "<<ax.transpose()<<" -> +z\n";
        cb = Vector3d::Zero(); ct = g.rot*(ct - g.origin); ax = Vector3d::UnitZ();
    }
    g.axis = ax;

    // boundary_flag from the CLEANED lists (the reader used the raw ones)
    mesh.boundary_flag.assign(n, 0);
    for(int i : g.top_nodes)    mesh.boundary_flag[i] = 1;
    for(int i : g.bottom_nodes) mesh.boundary_flag[i] = 1;

    //-------------------------------------------------------------------//
    // 3. boundary faces -> caps and two lateral components
    //-------------------------------------------------------------------//
    std::unordered_set<int> topset(g.top_nodes.begin(), g.top_nodes.end());
    std::unordered_set<int> botset(g.bottom_nodes.begin(), g.bottom_nodes.end());

    std::unordered_map<FaceKey, int, FaceKeyHash> count; count.reserve((size_t)m*2);
    std::unordered_map<FaceKey, BFace, FaceKeyHash> owner; owner.reserve((size_t)m*2);
    const int fl[4][3] = {{0,2,1},{0,1,3},{1,2,3},{0,3,2}};   // outward for a positive tet
    for(int e=0;e<m;e++){
        const std::vector<int>& c = mesh.elements[e];
        for(int f=0;f<4;f++){
            const FaceKey k = makeKey(c[fl[f][0]], c[fl[f][1]], c[fl[f][2]]);
            count[k]++;
            owner[k] = BFace{c[fl[f][0]], c[fl[f][1]], c[fl[f][2]], e};
        }
    }
    std::vector<BFace> lateral;
    int n_neg_vol = 0;
    for(const auto& kv : count){
        if(kv.second != 1) continue;
        BFace f = owner[kv.first];
        // orientation guard: flip if the normal points toward the opposite node
        const std::vector<int>& c = mesh.elements[f.elem];
        int opp = -1; for(int t=0;t<4;t++) if(c[t]!=f.a && c[t]!=f.b && c[t]!=f.c) opp = c[t];
        const Vector3d nrm = (mesh.nodes[f.b]-mesh.nodes[f.a]).cross(mesh.nodes[f.c]-mesh.nodes[f.a]);
        if(nrm.dot(mesh.nodes[opp]-mesh.nodes[f.a]) > 0.0){ std::swap(f.b, f.c); n_neg_vol++; }
        const bool cap = (topset.count(f.a) && topset.count(f.b) && topset.count(f.c))
                      || (botset.count(f.a) && botset.count(f.b) && botset.count(f.c));
        if(cap) g.cap_faces.push_back({f.a,f.b,f.c});
        else    lateral.push_back(f);
    }
    if(n_neg_vol) std::cout<<"  note: "<<n_neg_vol<<" boundary faces re-oriented (negative-volume tets)\n";

    // edge-connected components of the lateral faces
    {
        std::unordered_map<long long, std::vector<int>> edge2f; edge2f.reserve(lateral.size()*3);
        auto ekey = [](int i, int j){ if(i>j) std::swap(i,j); return ((long long)i<<32) | (unsigned)j; };
        for(size_t fi=0; fi<lateral.size(); fi++){
            const BFace& f = lateral[fi];
            edge2f[ekey(f.a,f.b)].push_back((int)fi);
            edge2f[ekey(f.b,f.c)].push_back((int)fi);
            edge2f[ekey(f.a,f.c)].push_back((int)fi);
        }
        DSU dsu((int)lateral.size());
        for(const auto& kv : edge2f) for(size_t t=1;t<kv.second.size();t++) dsu.unite(kv.second[0], kv.second[t]);
        std::unordered_map<int,int> comp_id; std::vector<int> comp_of(lateral.size());
        std::vector<double> comp_sign; std::vector<int> comp_size;
        for(size_t fi=0; fi<lateral.size(); fi++){
            const int r = dsu.find((int)fi);
            auto it = comp_id.find(r);
            if(it == comp_id.end()){ it = comp_id.emplace(r, (int)comp_sign.size()).first; comp_sign.push_back(0.0); comp_size.push_back(0); }
            comp_of[fi] = it->second; comp_size[it->second]++;
            const BFace& f = lateral[fi];
            const Vector3d cen = (mesh.nodes[f.a]+mesh.nodes[f.b]+mesh.nodes[f.c])/3.0;
            const Vector3d nrm = (mesh.nodes[f.b]-mesh.nodes[f.a]).cross(mesh.nodes[f.c]-mesh.nodes[f.a]).normalized();
            Vector3d rad = cen - cb; rad -= ax*rad.dot(ax);
            comp_sign[it->second] += nrm.dot(rad.normalized());
        }
        if(comp_sign.size() != 2){
            std::ostringstream e; e<<"exact_geometry: lateral surface splits into "<<comp_sign.size()
                                   <<" components (sizes:";
            for(int s : comp_size) e<<" "<<s;
            e<<"); expected exactly two (inner, outer). Check the end lists.";
            throw std::runtime_error(e.str());
        }
        const int outer_c = (comp_sign[0] > comp_sign[1]) ? 0 : 1;
        if(comp_sign[outer_c] <= 0.0 || comp_sign[1-outer_c] >= 0.0)
            throw std::runtime_error("exact_geometry: cannot tell inner from outer surface by normal direction");
        std::unordered_set<int> in_nodes, out_nodes;
        for(size_t fi=0; fi<lateral.size(); fi++){
            const BFace& f = lateral[fi];
            if(comp_of[fi] == outer_c){ g.outer_faces.push_back({f.a,f.b,f.c}); out_nodes.insert(f.a); out_nodes.insert(f.b); out_nodes.insert(f.c); }
            else                      { g.inner_faces.push_back({f.a,f.b,f.c}); in_nodes.insert(f.a);  in_nodes.insert(f.b);  in_nodes.insert(f.c);  }
        }
        g.inner_nodes.assign(in_nodes.begin(), in_nodes.end());  std::sort(g.inner_nodes.begin(), g.inner_nodes.end());
        g.outer_nodes.assign(out_nodes.begin(), out_nodes.end()); std::sort(g.outer_nodes.begin(), g.outer_nodes.end());
        int shared = 0; for(int i : g.inner_nodes) if(out_nodes.count(i)) shared++;
        std::cout<<"  lateral surface: "<<g.inner_faces.size()<<" inner faces ("<<g.inner_nodes.size()
                 <<" nodes), "<<g.outer_faces.size()<<" outer faces ("<<g.outer_nodes.size()
                 <<" nodes), "<<shared<<" shared; "<<g.cap_faces.size()<<" cap faces\n";
        if(shared) throw std::runtime_error("exact_geometry: inner and outer surfaces share nodes");
    }
    g.surface_code.assign(n, 0);
    for(int i : g.inner_nodes)  g.surface_code[i] = 1;
    for(int i : g.outer_nodes)  g.surface_code[i] = 2;
    for(int i : g.top_nodes)    g.surface_code[i] = 3;
    for(int i : g.bottom_nodes) g.surface_code[i] = 3;
    {
        int cnt[4] = {0,0,0,0}; for(int i=0;i<n;i++) cnt[g.surface_code[i]]++;
        std::cout<<"  nodes: "<<cnt[0]<<" interior, "<<cnt[1]<<" inner, "<<cnt[2]<<" outer, "<<cnt[3]<<" cap\n";
    }

    //-------------------------------------------------------------------//
    // 4. Laplace fields, frame, thickness
    //-------------------------------------------------------------------//
    std::vector<Grad4x3> G(m); std::vector<double> V(m);
    int n_inverted_mesh = 0;
    for(int e=0;e<m;e++){ tetGradient(mesh, e, G[e], V[e]); if(V[e] <= 0.0) n_inverted_mesh++; }
    if(n_inverted_mesh) std::cout<<"  *** WARNING: "<<n_inverted_mesh<<" tets have negative volume in the mesh ***\n";

    g.u = solveLaplace(mesh, G, V, g.inner_nodes, g.outer_nodes);
    g.v = solveLaplace(mesh, G, V, g.bottom_nodes, g.top_nodes);

    g.elem_n0.resize(m); g.elem_a0.resize(m); g.elem_s0.resize(m); g.elem_thickness.resize(m);
    for(int e=0;e<m;e++){
        const std::vector<int>& c = mesh.elements[e];
        Vector3d gu = Vector3d::Zero(), gv = Vector3d::Zero();
        for(int a=0;a<4;a++){ gu += G[e].row(a).transpose()*g.u[c[a]]; gv += G[e].row(a).transpose()*g.v[c[a]]; }
        const double gun = gu.norm();
        if(gun < 1e-14) throw std::runtime_error("exact_geometry: zero through-thickness gradient in an element");
        Vector3d n0 = gu/gun;
        Vector3d a0 = gv - gv.dot(n0)*n0;
        if(a0.norm() < 1e-10){
            // degenerate: fall back to the global axis projected on the tangent plane
            a0 = ax - ax.dot(n0)*n0;
            if(a0.norm() < 1e-10) a0 = n0.cross(Vector3d::UnitX());
        }
        a0.normalize();
        Vector3d s0 = n0.cross(a0); s0.normalize();
        g.elem_n0[e] = n0; g.elem_a0[e] = a0; g.elem_s0[e] = s0;
        g.elem_thickness[e] = 1.0/gun;
    }
    {
        std::vector<double> t = g.elem_thickness;
        std::cout<<"  frame built from two Laplace solves; thickness proxy 1/|grad u|:\n";
        printStat("thick", t, &V);
    }

    //-------------------------------------------------------------------//
    // 5. prestretch target
    //-------------------------------------------------------------------//
    const double lam_r = 1.0/(P.lam_z*P.lam_th);
    // closest points on both surfaces -> mid-surface offset
    double maxedge = 0.0;
    for(const auto& f : g.outer_faces) for(int t=0;t<3;t++)
        maxedge = std::max(maxedge, (mesh.nodes[f[t]]-mesh.nodes[f[(t+1)%3]]).norm());
    SurfaceGrid gin, gout;
    gin.build (mesh.nodes, g.inner_faces, std::max(0.5, 1.5*maxedge));
    gout.build(mesh.nodes, g.outer_faces, std::max(0.5, 1.5*maxedge));
    std::vector<Vector3d> corr(n);
    std::vector<double> thick_geo(n);
    for(int i=0;i<n;i++){
        double din, dout;
        const Vector3d pin  = gin.project (mesh.nodes[i], din);
        const Vector3d pout = gout.project(mesh.nodes[i], dout);
        corr[i] = mesh.nodes[i] - 0.5*(pin + pout);
        thick_geo[i] = din + dout;
    }
    std::cout<<"  wall thickness from closest points (nodal):\n";
    printStat("thick", thick_geo, nullptr);

    // smoothing along each surface, same-surface neighbours only
    auto smoothOn = [&](const std::vector<std::array<int,3>>& faces, const std::vector<int>& nodes){
        std::unordered_map<int, std::vector<int>> adj; adj.reserve(nodes.size());
        for(const auto& f : faces){
            adj[f[0]].push_back(f[1]); adj[f[0]].push_back(f[2]);
            adj[f[1]].push_back(f[0]); adj[f[1]].push_back(f[2]);
            adj[f[2]].push_back(f[0]); adj[f[2]].push_back(f[1]);
        }
        for(auto& kv : adj){ std::sort(kv.second.begin(), kv.second.end()); kv.second.erase(std::unique(kv.second.begin(), kv.second.end()), kv.second.end()); }
        for(int pass=0; pass<P.smooth_passes; pass++){
            std::vector<Vector3d> next = corr;
            for(int i : nodes){
                const std::vector<int>& nb = adj[i];
                if(nb.empty()) continue;
                Vector3d mean = Vector3d::Zero(); for(int j : nb) mean += corr[j];
                next[i] = 0.5*corr[i] + 0.5*mean/(double)nb.size();
            }
            corr.swap(next);
        }
    };
    smoothOn(g.inner_faces, g.inner_nodes);
    smoothOn(g.outer_faces, g.outer_nodes);

    // taper and assemble
    g.target.resize(n);
    for(int i=0;i<n;i++){
        const Vector3d& X = mesh.nodes[i];
        const double z  = (X - cb).dot(ax);
        const Vector3d Xp = (X - cb) - z*ax;
        const double beta = (P.taper > 0.0) ? smoothstep01(z/P.taper)*smoothstep01((g.length - z)/P.taper) : 1.0;
        g.target[i] = cb + P.lam_z*z*ax + P.lam_th*Xp + (lam_r - P.lam_th)*beta*corr[i];
    }

    // per-element check of the target map
    g.elem_theta_target.resize(m); g.elem_detF_target.resize(m); g.elem_lam_target.resize(m);
    int n_inv = 0, worst_e = -1; double worst_det = 1e30;
    for(int e=0;e<m;e++){
        const std::vector<int>& c = mesh.elements[e];
        Matrix3d F = Matrix3d::Zero();
        for(int a=0;a<4;a++) F += g.target[c[a]]*G[e].row(a);
        const double det = F.determinant();
        const Matrix3d C = F.transpose()*F;
        const Matrix3d cof = det*F.inverse().transpose();
        g.elem_detF_target[e]  = det;
        g.elem_theta_target[e] = (cof*g.elem_n0[e]).norm();
        g.elem_lam_target[e] = Vector3d(std::sqrt(g.elem_a0[e].dot(C*g.elem_a0[e])),
                                        std::sqrt(g.elem_s0[e].dot(C*g.elem_s0[e])),
                                        std::sqrt(g.elem_n0[e].dot(C*g.elem_n0[e])));
        if(det <= 0.0) n_inv++;
        if(det < worst_det){ worst_det = det; worst_e = e; }
    }
    std::cout<<"  prestretch target (lam_z "<<P.lam_z<<", lam_th "<<P.lam_th<<", lam_r "<<lam_r
             <<"; "<<P.smooth_passes<<" smoothing passes, taper "<<P.taper<<" mm), per element:\n";
    {
        std::vector<double> la(m), ls(m), ln(m);
        for(int e=0;e<m;e++){ la[e]=g.elem_lam_target[e](0); ls[e]=g.elem_lam_target[e](1); ln[e]=g.elem_lam_target[e](2); }
        printStat("theta_e", g.elem_theta_target, &V);
        printStat("detF",    g.elem_detF_target,  &V);
        printStat("lam_a",   la, &V);
        printStat("lam_s",   ls, &V);
        printStat("lam_n",   ln, &V);
    }
    if(n_inv){
        const std::vector<int>& c = mesh.elements[worst_e];
        std::ostringstream e;
        e<<"exact_geometry: the prestretch target inverts "<<n_inv<<" element(s); worst element "
         <<worst_e<<" det "<<worst_det<<" with node codes "<<g.surface_code[c[0]]<<g.surface_code[c[1]]
         <<g.surface_code[c[2]]<<g.surface_code[c[3]]<<". Raise WOUND_TAPER or lower WOUND_MIDSMOOTH.";
        throw std::runtime_error(e.str());
    }
    std::cout<<"===========================================================\n\n";
    return g;
}

//---------------------------------------------------------------------------//
void writeExactGeometryVTK(const HexMesh& mesh, const ExactGeometry& g,
                           const std::string& frame_file,
                           const std::string& target_file)
{
    const int n = mesh.n_nodes, m = mesh.n_elements;
    auto write = [&](const std::string& file, const std::vector<Vector3d>& pts, bool with_disp){
        std::ofstream f(file);
        if(!f.is_open()) throw std::runtime_error("cannot write " + file);
        f<<std::setprecision(10);
        f<<"# vtk DataFile Version 3.0\nexact dura geometry\nASCII\nDATASET UNSTRUCTURED_GRID\n";
        f<<"POINTS "<<n<<" double\n";
        for(int i=0;i<n;i++) f<<pts[i](0)<<" "<<pts[i](1)<<" "<<pts[i](2)<<"\n";
        f<<"CELLS "<<m<<" "<<5*m<<"\n";
        for(int e=0;e<m;e++) f<<"4 "<<mesh.elements[e][0]<<" "<<mesh.elements[e][1]<<" "<<mesh.elements[e][2]<<" "<<mesh.elements[e][3]<<"\n";
        f<<"CELL_TYPES "<<m<<"\n"; for(int e=0;e<m;e++) f<<"10\n";
        f<<"CELL_DATA "<<m<<"\n";
        auto vec = [&](const char* name, const std::vector<Vector3d>& a){
            f<<"VECTORS "<<name<<" double\n"; for(int e=0;e<m;e++) f<<a[e](0)<<" "<<a[e](1)<<" "<<a[e](2)<<"\n"; };
        auto sca = [&](const char* name, const std::vector<double>& a){
            f<<"SCALARS "<<name<<" double 1\nLOOKUP_TABLE default\n"; for(int e=0;e<m;e++) f<<a[e]<<"\n"; };
        vec("a0", g.elem_a0); vec("s0", g.elem_s0); vec("n0", g.elem_n0);
        std::vector<double> ue(m), ve(m);
        for(int e=0;e<m;e++){ ue[e]=0; ve[e]=0; for(int a=0;a<4;a++){ ue[e]+=0.25*g.u[mesh.elements[e][a]]; ve[e]+=0.25*g.v[mesh.elements[e][a]]; } }
        sca("u", ue); sca("v", ve); sca("thickness", g.elem_thickness);
        sca("theta_e_target", g.elem_theta_target); sca("detF_target", g.elem_detF_target);
        f<<"SCALARS tag int 1\nLOOKUP_TABLE default\n";
        for(int e=0;e<m;e++) f<<(mesh.element_domain_id.empty() ? 0 : mesh.element_domain_id[e])<<"\n";
        f<<"POINT_DATA "<<n<<"\n";
        f<<"SCALARS surface int 1\nLOOKUP_TABLE default\n"; for(int i=0;i<n;i++) f<<g.surface_code[i]<<"\n";
        f<<"SCALARS u double 1\nLOOKUP_TABLE default\n"; for(int i=0;i<n;i++) f<<g.u[i]<<"\n";
        f<<"SCALARS v double 1\nLOOKUP_TABLE default\n"; for(int i=0;i<n;i++) f<<g.v[i]<<"\n";
        if(with_disp){
            f<<"VECTORS prestretch_disp double\n";
            for(int i=0;i<n;i++){ const Vector3d d = g.target[i]-mesh.nodes[i]; f<<d(0)<<" "<<d(1)<<" "<<d(2)<<"\n"; }
        }
    };
    write(frame_file, mesh.nodes, true);
    write(target_file, g.target, false);
    std::cout<<"wrote "<<frame_file<<" and "<<target_file<<"\n";
}
