/**
 * @file sphiros.cpp
 * @brief Main entry point for the SPHiros application.
 *
 * This file initializes the MPI and Kokkos runtimes, parses command-line
 * arguments, and demonstrates the usage of various EOS (Equation of State)
 * implementations. It also includes examples of using Kokkos views and
 * performing computations with EOS objects.
 */

// std includes
#include <iostream>
#include <vector>
#include <typeinfo>
// mpi includes
#include <mpi.h>
// kokkos includes
#include <Kokkos_Core.hpp>
// yaml-cpp includes
#include <yaml-cpp/yaml.h>
// CLI includes
#include <CLI/CLI.hpp>
// eos includes (test-only)
#include "eos_linear_gas.hpp"
#include "eos_stiffened_gas.hpp"

/**
 * @brief Main function for the SPHiros application.
 *
 * Initializes MPI and Kokkos runtimes, parses command-line arguments, and
 * demonstrates the usage of EOS objects with Kokkos views.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return int Exit code of the application.
 */
int main(int argc, char** argv) {
    // Initialize the MPI runtime
    MPI_Init(&argc, &argv);
    // Initialize the Kokkos runtime
    Kokkos::initialize(argc, argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    CLI::App app{"SPHiros Command-Line Parser"};
    std::string input_file;
    app.add_option("-i,--input", input_file, "Input YAML file")->required();
    std::string output_file;
    app.add_option("-o,--output", output_file, "Output file");
    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "Enable verbose output");

    // Parse command-line arguments
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        // Finalize the Kokkos runtime
        Kokkos::finalize();
        MPI_Finalize();
        return app.exit(e);
    }

    if (world_rank == 0) {
        // Display parsed values
        std::cout << "Input File: " << input_file << std::endl;
        std::cout << "Output File: " << output_file << std::endl;
        std::cout << "Verbose: " << (verbose ? "Enabled" : "Disabled")
                  << std::endl;
    }

    {
        // Create a container of std::variant to hold EOSLinearGas and
        // EOSStiffenedGas
        using EOSVariant = std::variant<EOSLinearGas, EOSStiffenedGas>;
        std::vector<EOSVariant> eos_vector;

        // Add EOSLinearGas and EOSStiffenedGas instances to the container
        eos_vector.emplace_back(EOSLinearGas(0, 1.4, 1e-6));
        eos_vector.emplace_back(EOSStiffenedGas(1, 1.4, 1e-6, 0.0));
        eos_vector.emplace_back(EOSLinearGas(2, 1.4, 1e-6));
        eos_vector.emplace_back(EOSLinearGas(3, 1.4, 1e-6));

        // Example Kokkos views
        Kokkos::View<double*> rho("rho", 10);
        Kokkos::View<double*> eint("eint", 10);
        Kokkos::View<double*> p("p", 10);
        Kokkos::View<double*> sos("sos", 10);

        // Copy data from host to device
        Kokkos::deep_copy(rho, 1.0);
        Kokkos::deep_copy(eint, 1.0);

        // Call PressureSoS for each EOS object using std::visit
        for (const auto& eos : eos_vector) {
            std::visit(
                [&](const auto& obj) {
                    std::string type_name = typeid(obj).name();
                    if (type_name.length() > 20) {
                        type_name = type_name.substr(0, 20) + "...";
                    }
                    std::cout << "EOS Type: " << type_name << std::endl;
                    obj.PressureSoSImpl(rho, eint, p, sos);
                    auto host_p   = Kokkos::create_mirror_view(p);
                    auto host_sos = Kokkos::create_mirror_view(p);

                    Kokkos::deep_copy(host_p, p);
                    Kokkos::deep_copy(host_sos, sos);

                    std::cout << "p: " << host_p(0) << std::endl;
                    std::cout << "sos: " << host_sos(0) << std::endl;
                },
                eos);
        }
    }

    // Finalize the Kokkos runtime
    Kokkos::finalize();
    // Finalize the MPI runtime
    MPI_Finalize();

    return 0;
}