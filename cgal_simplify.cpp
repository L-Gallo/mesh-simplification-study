// cgal_simplify.cpp
// CGAL-based mesh simplification command-line tool
// Compatible with CGAL 6.1+ and Python mesh simplification comparison framework
//
// Usage: cgal_simplify <input.ply> <output.ply> <keep_ratio> [<output_stats.json>]
//   keep_ratio: fraction of triangles to keep (e.g., 0.25 for 75% reduction)
//
// Compile:
//   mkdir build && cd build
//   cmake ..
//   cmake --build . --config Release

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_count_ratio_stop_predicate.h>

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <cmath>

// Type definitions
typedef CGAL::Simple_cartesian<double>                       Kernel;
typedef Kernel::Point_3                                      Point_3;
typedef CGAL::Surface_mesh<Point_3>                          Surface_mesh;

namespace SMS = CGAL::Surface_mesh_simplification;

// JSON output helper
void write_json_stats(const std::string& filepath,
                      size_t input_vertices,
                      size_t input_faces,
                      size_t output_vertices,
                      size_t output_faces,
                      double execution_time_ms,
                      double target_ratio,
                      double actual_ratio)
{
    std::ofstream out(filepath);
    if (!out) {
        std::cerr << "Warning: Could not write stats to " << filepath << std::endl;
        return;
    }

    out << "{\n";
    out << "  \"input_vertices\": " << input_vertices << ",\n";
    out << "  \"input_faces\": " << input_faces << ",\n";
    out << "  \"output_vertices\": " << output_vertices << ",\n";
    out << "  \"output_faces\": " << output_faces << ",\n";
    out << "  \"execution_time_ms\": " << execution_time_ms << ",\n";
    out << "  \"target_reduction_ratio\": " << (1.0 - target_ratio) << ",\n";
    out << "  \"actual_reduction_ratio\": " << (1.0 - actual_ratio) << ",\n";
    out << "  \"target_keep_ratio\": " << target_ratio << ",\n";
    out << "  \"actual_keep_ratio\": " << actual_ratio << "\n";
    out << "}\n";

    out.close();
}

int main(int argc, char* argv[])
{
    // Parse arguments
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_mesh> <output_mesh> <keep_ratio> [<output_stats.json>]\n";
        std::cerr << "\n";
        std::cerr << "Arguments:\n";
        std::cerr << "  input_mesh     : Input mesh file (.ply, .obj, .off)\n";
        std::cerr << "  output_mesh    : Output mesh file (.ply, .obj, .off)\n";
        std::cerr << "  keep_ratio     : Fraction of faces to keep (0.0-1.0)\n";
        std::cerr << "                   e.g., 0.25 means keep 25% (75% reduction)\n";
        std::cerr << "  output_stats   : Optional JSON file for statistics\n";
        std::cerr << "\n";
        std::cerr << "Supported formats: PLY, OBJ, OFF\n";
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    double keep_ratio = std::atof(argv[3]);
    std::string stats_file = (argc >= 5) ? argv[4] : "";

    // Validate keep_ratio
    if (keep_ratio <= 0.0 || keep_ratio > 1.0) {
        std::cerr << "Error: keep_ratio must be between 0.0 and 1.0 (got " << keep_ratio << ")\n";
        return 1;
    }

    // Load mesh
    Surface_mesh mesh;
    std::cout << "Loading mesh: " << input_file << std::endl;

    std::ifstream input(input_file);
    if (!input) {
        std::cerr << "Error: Cannot open input file: " << input_file << std::endl;
        return 1;
    }

    // Detect format and load
    bool load_success = false;
    if (input_file.find(".ply") != std::string::npos) {
        load_success = CGAL::IO::read_PLY(input, mesh);
    } else if (input_file.find(".off") != std::string::npos) {
        load_success = CGAL::IO::read_OFF(input, mesh);
    } else if (input_file.find(".obj") != std::string::npos) {
        load_success = CGAL::IO::read_OBJ(input, mesh);
    } else {
        std::cerr << "Error: Unsupported file format. Use .ply, .obj, or .off\n";
        return 1;
    }

    input.close();

    if (!load_success || mesh.is_empty()) {
        std::cerr << "Error: Failed to load mesh or mesh is empty\n";
        return 1;
    }

    // Get input statistics
    size_t input_vertices = mesh.number_of_vertices();
    size_t input_faces = mesh.number_of_faces();

    std::cout << "Input mesh: " << input_vertices << " vertices, " 
              << input_faces << " faces" << std::endl;

    // Validate mesh
    if (!CGAL::is_triangle_mesh(mesh)) {
        std::cerr << "Error: Input mesh is not a pure triangle mesh\n";
        return 1;
    }

    // Simplification
    std::cout << "Simplifying to " << (keep_ratio * 100) << "% of original faces..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Create stop predicate (keep_ratio of faces)
    // CGAL 6.1 uses Edge_count_ratio_stop_predicate
    SMS::Edge_count_ratio_stop_predicate<Surface_mesh> stop(keep_ratio);

    // Simplify using default Lindstrom-Turk method
    // CGAL 6.1 doesn't need explicit index maps - they're automatic
    int removed_edges = SMS::edge_collapse(mesh, stop);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double execution_time_ms = duration.count() / 1000.0;

    // Get output statistics
    size_t output_vertices = mesh.number_of_vertices();
    size_t output_faces = mesh.number_of_faces();
    double actual_ratio = static_cast<double>(output_faces) / static_cast<double>(input_faces);

    std::cout << "Removed " << removed_edges << " edges" << std::endl;
    std::cout << "Output mesh: " << output_vertices << " vertices, " 
              << output_faces << " faces" << std::endl;
    std::cout << "Actual reduction: " << ((1.0 - actual_ratio) * 100) << "%" << std::endl;
    std::cout << "Execution time: " << execution_time_ms << " ms" << std::endl;

    // Save simplified mesh
    std::cout << "Saving mesh: " << output_file << std::endl;

    std::ofstream output(output_file);
    if (!output) {
        std::cerr << "Error: Cannot create output file: " << output_file << std::endl;
        return 1;
    }

    bool save_success = false;
    if (output_file.find(".ply") != std::string::npos) {
        save_success = CGAL::IO::write_PLY(output, mesh);
    } else if (output_file.find(".off") != std::string::npos) {
        save_success = CGAL::IO::write_OFF(output, mesh);
    } else if (output_file.find(".obj") != std::string::npos) {
        save_success = CGAL::IO::write_OBJ(output, mesh);
    }

    output.close();

    if (!save_success) {
        std::cerr << "Error: Failed to save mesh\n";
        return 1;
    }

    // Write statistics JSON if requested
    if (!stats_file.empty()) {
        write_json_stats(stats_file, input_vertices, input_faces,
                        output_vertices, output_faces, execution_time_ms,
                        keep_ratio, actual_ratio);
        std::cout << "Statistics saved: " << stats_file << std::endl;
    }

    std::cout << "Success!" << std::endl;
    return 0;
}