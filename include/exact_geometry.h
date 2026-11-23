/*
  Geometry pre-processing for a REAL (scanned, non-cylindrical) dura mesh.

  The cylinder drivers get three things for free from the analytic geometry:
    1. which nodes lie on the inner / outer lateral surfaces and on the end caps,
    2. the fibre frame a0 (axial), s0 (circumferential), n0 (through-thickness),
    3. the physiological prestretch target x(X) with axial 1.098, circumferential
       1.035 and through-thickness 1/(1.098*1.035) = 0.880 about the mid-surface.
  On a scanned sac none of these has a closed form. This module derives all three
  from the tet mesh itself, plus the two end-cap node lists that come with it:

    surfaces : boundary faces (faces used by exactly one tet); cap faces have all
               three nodes in one end list; the remaining lateral faces split into
               exactly TWO edge-connected components, labelled inner/outer by the sign
               of outward_normal . radial_from_axis averaged over the component.
    frame    : two Laplace problems on the tets. u: inner 0 -> outer 1 gives
               n0 = grad u / |grad u|; v: bottom 0 -> top 1 gives a0 = grad v projected
               onto the tangent plane; s0 = n0 x a0 (same handedness as
               build_cylinder_frame in the cylinder drivers).
    target   : x = A(X) + (lam_r - lam_th) * beta(z) * corr(X)
               A      = affine (axial lam_z, in-plane homothety lam_th) - exact areal
                        stretch lam_z*lam_th on its own,
               corr   = offset of X from the mid-surface, the mid-surface being the
                        midpoint of the closest points on the inner and outer surfaces,
                        smoothed along each surface (nodal-normal versions invert the
                        micron-thin slivers at the cap rims; this one does not),
               beta   = smoothstep taper to 0 within `taper` mm of each cap, so the map
                        is purely affine where the slivers are.
               Measured on dura_only_double_wound_4t: theta_e 1.1227/1.1493 at the
               1st/99th percentile, matching the cylinder's own 1.1254/1.1462.

  The mesh is also ALIGNED in place: bottom-cap centroid to the origin, bottom->top
  axis onto +z. A rigid motion, deterministic on restart, and it puts the geometry in
  the cylinder conventions (z in [0, L], wound near mid-length).

  The end lists as shipped are 0-based and may contain a few stray ids (the 4t top
  list has two that sit on the bottom cap). They are sanitised by axial coordinate
  before anything else, and boundary_flag is rewritten from the cleaned sets.
*/

#ifndef exact_geometry_h
#define exact_geometry_h

#include <array>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "myMeshGenerator.h"

struct ExactGeomParams {
    double lam_z         = 1.098;   // axial prestretch
    double lam_th        = 1.035;   // circumferential prestretch
    int    smooth_passes = 10;      // Jacobi passes on the mid-surface offset, per surface
    double taper         = 2.0;     // [mm] through-thickness correction fades to 0 this close to a cap
    double list_tol      = 1.0;     // [mm] an end-list id further than this from its list's median axial coordinate is dropped
    bool   align         = true;    // move the mesh into the aligned frame
};

struct ExactGeometry {
    // --- node classification (indices into mesh.nodes) ---
    std::vector<int> top_nodes, bottom_nodes;          // cleaned end lists
    std::vector<int> dropped_top, dropped_bottom;      // stray ids that were removed
    std::vector<int> inner_nodes, outer_nodes;
    std::vector<std::array<int,3>> inner_faces, outer_faces, cap_faces;
    std::vector<int> surface_code;                     // per node: 0 interior, 1 inner, 2 outer, 3 cap

    // --- alignment applied to the mesh (x_aligned = rot*(x_scan - origin)) ---
    Eigen::Vector3d origin;
    Eigen::Matrix3d rot;
    double length;                                     // cap-centroid to cap-centroid, after alignment = z extent
    Eigen::Vector3d axis;                              // in the aligned frame this is (0,0,1)

    // --- Laplace fields (nodal) ---
    std::vector<double> u;                             // 0 inner -> 1 outer
    std::vector<double> v;                             // 0 bottom -> 1 top

    // --- per-element frame and thickness ---
    std::vector<Eigen::Vector3d> elem_n0, elem_a0, elem_s0;
    std::vector<double> elem_thickness;                // 1/|grad u|

    // --- prestretch target ---
    std::vector<Eigen::Vector3d> target;               // per node
    std::vector<double> elem_theta_target;             // ||cof(F).n0|| of the target map, per element
    std::vector<double> elem_detF_target;
    std::vector<Eigen::Vector3d> elem_lam_target;      // (lam_a, lam_s, lam_n) of the target map
};

// Builds everything above. MODIFIES the mesh: nodes are aligned (if params.align) and
// boundary_flag is rewritten from the cleaned end lists (1 on both caps, 0 elsewhere).
// Throws if the lateral surface does not split into exactly two components, if a list
// loses more than 1 % of its ids, or if the target map inverts any element.
ExactGeometry buildExactGeometry(HexMesh& mesh,
                                 const std::string& bottom_list_file,
                                 const std::string& top_list_file,
                                 const ExactGeomParams& params);

// Legacy VTK writers for inspection in ParaView.
//   frame_file : points at the (aligned) reference, cell data a0/s0/n0/u/v/thickness/tag/
//                theta_e_target/detF_target, point data surface/u/v/prestretch_disp.
//   target_file: same cell data, points at the target positions.
void writeExactGeometryVTK(const HexMesh& mesh, const ExactGeometry& g,
                           const std::string& frame_file,
                           const std::string& target_file);

// Percentile helper shared with the driver's reports (p in [0,100]).
double percentile(std::vector<double> v, double p);

#endif
