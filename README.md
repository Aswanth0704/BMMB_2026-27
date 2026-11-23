# BMMB_2026-27
Spinal Dura Mater Puncture Healing Finite Element Code

A three-dimensional finite element model of wound healing in the spinal dura mater. It couples large-deformation mechanics of the prestretched, fiber-reinforced dura with reaction-diffusion of a pro-inflammatory signal, TGF-β and fibroblasts, and with the local evolution of collagen fraction, fiber orientation, fiber dispersion and plastic growth. All biochemical fields are normalized so that healthy tissue sits at 1.

The code has been designed to run on Purdue Research Computing Clusters (i.e., Negishi Community Cluster). It may not work as-is in your system. Here are a few tips to troubleshoot compiling:
<ul>
<li>The code requires the EIGEN linear algebra library and the (header-only) BOOST libraries. It currently runs with BOOST 1.80 and GCC 12.2</li>
<li>The code has a parallel version with OPEN MP. The parallel parts of the code are primarily in solver.cpp</li>
<li>Compilation is done with CMAKE (3.11 or newer, C++17)</li>
</ul>

## Compiling
<ul>
<li>CMake looks for Eigen and Boost next to the project folder, then in the usual system locations. Otherwise pass <code>-DEIGEN_INCLUDE_DIR=&lt;dir containing Eigen/&gt;</code> and <code>-DBOOST_INCLUDE_DIR=&lt;dir containing boost/&gt;</code></li>
<li>On Negishi: <code>module load gcc/12.2.0 boost/1.80.0</code></li>
<li><code>mkdir build && cd build && cmake -DUSE_MKL=OFF .. && make -j12</code></li>
<li>On macOS with Apple clang, install OpenMP with <code>brew install libomp</code> and add <code>-DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_omp_LIBRARY=$(brew --prefix libomp)/lib/libomp.dylib</code> to the cmake command</li>
<li>This builds the executable <code>woundcpp3D</code>. Choose which simulation it runs with <code>-DMAIN_SRC</code> (see below)</li>
</ul>

## Simulations
| Driver (`-DMAIN_SRC=`) | Simulation | Mesh files (in `files/`) |
|---|---|---|
| `src/cydindrical_dura_single_tissue_multiwound.cpp` (default) | Single 25-gauge puncture in the idealized cylindrical dura | `dura_cyl_repeated_wound_v62_2t_finer.mphtxt` |
| `src/dura_cyl_two_wound_staged.cpp` | Two sequential punctures in the cylindrical dura | `dura_cyl_repeated_wound_v62_{2,4,6,8,10}t_finer.mphtxt` (second wound at 2, 4, 6, 8 or 10 dural thicknesses from the first) |
| `src/dura_exact_single_wound.cpp` | Single puncture in the anatomical L3-L5 dural sac | `exact_dura/dura_only_double_wound_4t.msh` with `top_end_nodes_double_wound_4t.txt` and `bottom_end_nodes_double_wound_4t.txt` |

Each run first settles the healthy tissue into its physiological prestretch, then makes the puncture and follows healing.

## Running
The executable reads mesh files from, and writes all output to, the current directory, so give every case its own run directory:
```bash
mkdir run_case && cd run_case
cp ../build/woundcpp3D ../files/woundcpp3D.sub ../files/dura_cyl_repeated_wound_v62_4t_finer.mphtxt .
WOUND_MESH=dura_cyl_repeated_wound_v62_4t_finer.mphtxt sbatch woundcpp3D.sub
```
Edit the account and modules in `woundcpp3D.sub` for your cluster. For the anatomical geometry, copy the `.msh` file and both node lists from `files/exact_dura/`.

Common options are set with environment variables, no rebuild needed:
| Variable | Meaning |
|---|---|
| `WOUND_MESH` | mesh file name |
| `WOUND_TSETTLE` | prestretch settling time before the puncture [h] (default 100) |
| `WOUND_TFINAL` | healing time after the puncture [h] (single-wound drivers) |
| `WOUND_SEP` | separation of the second wound, in dural thicknesses (two-wound driver, default 2) |
| `WOUND_DELAY` | weeks of healing before the second puncture (two-wound driver, default 1) |
| `WOUND_TAFTER` | healing time after the second puncture [h] (two-wound driver, default 672) |
| `WOUND_OUT` | output file prefix |
| `WOUND_RESTART` | absolute path of a checkpoint `.txt` to resume from, using the same settings as the original run |

## Output
Legacy VTK files (`<prefix><step>.vtk` and `<prefix>second_<step>.vtk`) with fibroblast density, TGF-β, collagen fraction, fiber dispersion, pro-inflammatory signal, fiber direction, strain, stress and plastic stretch, readable in ParaView. A checkpoint `<prefix><step>.txt` is written with every VTK save and can be used with `WOUND_RESTART`.
