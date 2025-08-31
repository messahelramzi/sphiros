#pragma once

#include <pybind11/embed.h>  // For embedding Python
#include <iostream>

namespace py = pybind11;

void use_meshio(const std::string& input_file, const std::string& output_file) {
    try {
        pybind11::scoped_interpreter guard{};  // Start the Python interpreter

        // Import the meshio library
        pybind11::module meshio = pybind11::module::import("meshio");

        // Read the mesh file
        pybind11::object mesh = meshio.attr("read")(input_file);

        // Write the mesh to a new file
        meshio.attr("write")(output_file, mesh);

        std::cout << "Mesh successfully read from " << input_file
                  << " and written to " << output_file << std::endl;
    } catch (const pybind11::error_already_set& e) {
        std::cerr << "Python error: " << e.what() << std::endl;
    }
}