# SPHiros Project

This project implements a Smoothed Particle Hydrodynamics (SPH) hydrocode designed for high-performance computing (HPC) environments. It leverages Kokkos to achieve portability and efficient execution on diverse hardware architectures (CPUs, GPUs, APUs, *etc*.). The code supports hybrid parallelism using MPI for distributed memory and Kokkos for on-node parallelism, enabling scalable simulations across diverse hardware architectures.

## Project Structure

```
sphiros
├── .github/           
│   ├── workflows      
│   │   ├── deploy.yml      
├── config/                       # Spack recipies and other config files
├── examples/                     # An example folder: Mesh and input files
│   ├── sod.yaml                  # Sod shock tube example
├── src/                          # Source code directory
│   ├── eos/                      # Equations of states
│   │   ├── CMakeLists.txt
│   │   ├── eos_crtp.hpp          # CRTP base class for equation of state (EOS) implementations
│   │   ├── eos_linear_gas.hpp    # Linear Gas EOS
│   │   ├── eos_stiffened_gas.hpp # Stiffened Gas EOS
│   ├── material_models/          # Material models: Naviers-Stokes, Elastic, Elasto-Plastic, *etc*.0
│   ├── spatial_solvers/          # SPH spatial approximations
│   ├── time_integrators/         # Time integrators: Leap-Frog
│   ├── CMakeLists.txt    
│   └── sphiros.cpp               # Main application entry point
├── tests/                        # Unitary and regressions tests using Google Test framework
│   ├── eos/                      # Equations of states
│   │   ├── CMakeLists.txt
│   │   ├── eos_linear_gas.cpp    # Linear Gas EOS
│   │   ├── eos_stiffened_gas.cpp # Stiffened Gas EOS
│   └── CMakeLists.txt    
├── .clang-format
├── .clang-tidy 
├── .gitignore
├── CMakeLists.txt        # CMake build configuration
├── CMakePresets.json
├── CONTRIBUTING.md
├── Doxyfile 
├── README.md             # Project documentation
└── LICENSE               # License information
```

## Requirements

- C++20 compatible compiler (GCC & CLANG were tested so far)
- MPI library
- Kokkos with OpenMP support and accelerator support if necessary (Cuda, HIP, *etc*.)
- Spdlog
- Yaml-CPP
- CLI11
- CMake
- ArborX
- PDI for IOs and relative plugins (HDF5, MPI & Pycall)
- Doxygen to build the documentation
- Lcov to generate coverage reports

## Building the Project

To build the project, follow these steps:

0. Pre-requirement: Install dev environment using Spack
   ```bash
   git clone --depth=2 https://github.com/spack/spack.git
   . spack/share/spack/setup-env.sh
   spack env create sphiros sphiros/spack-[openmp_or_cuda].yam
   spack install
   ```

1. Navigate to the project directory:

   ```bash
   cd sphiros
   ```

2. Create a build directory:

   ```bash
   mkdir build
   cd build
   ```

3. Run CMake to configure the project:

   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \ -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    -S ../
   ```

4. Build the project:

   ```bash
   make -j
   ```

## Running the Application

To run the application using MPI, use the following command:

```bash
mpirun -np <number_of_processes> path-to-execs/Sphiros -i path-to-input.yaml
```

Replace `<number_of_processes>` with the desired number of MPI processes and `your_executable_name` with the name of the compiled executable. 

Specify the number of OpenMP threads by defining the environment variable:
```bash
export OMP_NUM_THREADS=`<number_of_threads_per_process>`
```

For GPU applications, we consider one GPU per MPI rank and affinity should be automatically set by Kokkos and hwloc.

To print the configuration, add the option *--kokkos-print-configuration* to the execution command-line. For more details, please visit [Kokkos documentation](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Initialization.html).

## Building the documentation

From sphiros root after CMake configuration:
```bash
cmake --build build --target doc --
```

## Building coverage report

From sphiros root after CMake configuration:
```bash
bash ./scripts/generate_coverage.sh
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
