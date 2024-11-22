#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>

// Assimp for loading STL files
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// Eigen for linear algebra
#include <Eigen/Dense>

// OpenCV for image creation and saving
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// Structure to hold mesh data
struct Mesh {
    Eigen::MatrixXd vertices; // Nx3 matrix
    Eigen::MatrixXd colors;   // Nx3 matrix
};

// Function to load STL file using Assimp and convert to Eigen matrices
Mesh loadSTL(const std::string& filepath) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filepath, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);

    if (!scene || !scene->HasMeshes()) {
        throw std::runtime_error("Failed to load mesh: " + filepath);
    }

    const aiMesh* ai_mesh = scene->mMeshes[0];
    Mesh mesh;
    mesh.vertices.resize(ai_mesh->mNumVertices, 3);
    mesh.colors.resize(ai_mesh->mNumVertices, 3);

    // Load vertices
    for (unsigned int i = 0; i < ai_mesh->mNumVertices; ++i) {
        mesh.vertices(i, 0) = ai_mesh->mVertices[i].x;
        mesh.vertices(i, 1) = ai_mesh->mVertices[i].y;
        mesh.vertices(i, 2) = ai_mesh->mVertices[i].z;
    }

    // Load colors if available, else assign default color
    if (ai_mesh->HasVertexColors(0)) {
        for (unsigned int i = 0; i < ai_mesh->mNumVertices; ++i) {
            mesh.colors(i, 0) = ai_mesh->mColors[0][i].r;
            mesh.colors(i, 1) = ai_mesh->mColors[0][i].g;
            mesh.colors(i, 2) = ai_mesh->mColors[0][i].b;
        }
    } else {
        mesh.colors = Eigen::MatrixXd::Ones(ai_mesh->mNumVertices, 3);
    }

    return mesh;
}

// Function to compute center of mass
Eigen::Vector3d computeCenterOfMass(const Eigen::MatrixXd& vertices) {
    return vertices.colwise().mean();
}

// Function to normalize vertices to unit sphere
Eigen::MatrixXd normalizeToUnitSphere(const Eigen::MatrixXd& vertices) {
    Eigen::MatrixXd normalized = vertices;
    for (int i = 0; i < normalized.rows(); ++i) {
        double norm = normalized.row(i).norm();
        if (norm > 0) {
            normalized.row(i) /= norm;
        }
    }
    return normalized;
}

// Function to normalize colors based on vertices
Eigen::MatrixXd normalizeColors(const Eigen::MatrixXd& vertices) {
    Eigen::MatrixXd normalized = vertices;
    Eigen::Vector3d min = vertices.colwise().minCoeff();
    Eigen::Vector3d max = vertices.colwise().maxCoeff();
    Eigen::Vector3d range = max - min;
    for (int i = 0; i < normalized.rows(); ++i) {
        for (int j = 0; j < 3; ++j) {
            if (range(j) != 0)
                normalized(i, j) = (normalized(i, j) - min(j)) / range(j);
            else
                normalized(i, j) = 0.0;
        }
    }
    return normalized;
}

void saveGrayscaleProjection(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& intensity, const std::string& output_path, int width = 400, int height = 400, int radius = 1) {
    // Create a grayscale image with a white background
    cv::Mat image = cv::Mat::ones(height, width, CV_8UC1) * 255;

    // Normalize x, y, and intensity to fit within image bounds and grayscale range
    Eigen::VectorXd x_normalized = (x.array() - x.minCoeff()) / (x.maxCoeff() - x.minCoeff()) * (width - 1);
    Eigen::VectorXd y_normalized = (y.array() - y.minCoeff()) / (y.maxCoeff() - y.minCoeff()) * (height - 1);
    Eigen::VectorXd intensity_normalized = (intensity.array() - intensity.minCoeff()) / (intensity.maxCoeff() - intensity.minCoeff()) * 255;

    // Plot each point with a grayscale circle based on the third coordinate
    for (int i = 0; i < x.size(); ++i) {
        int img_x = static_cast<int>(x_normalized(i));
        int img_y = static_cast<int>(y_normalized(i));
        uchar grayscale_value = static_cast<uchar>(255 - intensity_normalized(i)); // Invert for correct grayscale intensity
        
        // Draw a filled circle for each point
        cv::circle(image, cv::Point(img_x, img_y), radius, cv::Scalar(grayscale_value), -1); // -1 for filled circle
    }

    // Save the image
    cv::imwrite(output_path, image);
}

// Function to create directory if it doesn't exist
void createDirectory(const std::string& path) {
    if (!fs::exists(path)) {
        fs::create_directories(path);
    }
}

void mapToImage(const Eigen::VectorXd& u, const Eigen::VectorXd& v, 
               const Eigen::MatrixXd& colors, cv::Mat& image, 
               const cv::Scalar& colorMultiplier, int radius = 3) {
    // Assuming u and v are in [0,1]
    int width = image.cols;
    int height = image.rows;
    
    for (int i = 0; i < u.size(); ++i) {
        int x = static_cast<int>(u(i) * (width - 1));
        int y = static_cast<int>((1.0 - v(i)) * (height - 1)); // Flip y for image coordinates

        // Clamp coordinates to be within the image bounds
        x = std::min(std::max(x, 0), width - 1);
        y = std::min(std::max(y, 0), height - 1);

        // Scale colors from [0,1] to [0,255]
        uchar r = static_cast<uchar>(std::round(colors(i, 0) * 255.0));
        uchar g = static_cast<uchar>(std::round(colors(i, 1) * 255.0));
        uchar b = static_cast<uchar>(std::round(colors(i, 2) * 255.0));

        // Draw a circle at the specified coordinates with the color
        cv::circle(image, cv::Point(x, y), radius, cv::Scalar(b, g, r), -1); // -1 for filled circle
    }
}


// ThreadPool class definition (as provided earlier)
class ThreadPool {
public:
    ThreadPool(size_t num_threads);
    ~ThreadPool();

    // Enqueue a task
    void enqueue(std::function<void()> task);

    // Wait for all tasks to complete
    void wait_until_empty();

private:
    // Worker function for each thread
    void worker();

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

    std::mutex wait_mutex;
    std::condition_variable wait_condition;
    std::atomic<size_t> tasks_in_progress;
};

ThreadPool::ThreadPool(size_t num_threads) : stop(false), tasks_in_progress(0) {
    for (size_t i = 0; i < num_threads; ++i)
        workers.emplace_back(&ThreadPool::worker, this);
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.emplace(task);
    }
    condition.notify_one();
}

void ThreadPool::wait_until_empty() {
    std::unique_lock<std::mutex> lock(wait_mutex);
    wait_condition.wait(lock, [this]() { return tasks.empty() && tasks_in_progress == 0; });
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread &worker: workers)
        worker.join();
}

void ThreadPool::worker() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(lock, [this]() { return stop || !tasks.empty(); });
            if (stop && tasks.empty())
                return;
            task = std::move(tasks.front());
            tasks.pop();
            tasks_in_progress++;
        }
        task();
        tasks_in_progress--;
        wait_condition.notify_all();
    }
}

// Main processing function with multithreading
int main(int argc, char* argv[]) {
    try {
        // Check if the correct number of arguments is provided
        

        // Read input and output directories from command-line arguments
        std::string input_dir = argv[1];
        std::string output_dir = argv[2];
        int radius = std::stoi(argv[3]);

        // Create output directory if it doesn't exist
        createDirectory(output_dir);

        // Gather all relevant STL files first
        std::vector<fs::path> stl_files;
        for (const auto& entry : fs::recursive_directory_iterator(input_dir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename.find("stl") != std::string::npos &&
                    filename.size() >= 4 &&
                    filename.substr(filename.size() - 4) == ".stl") {
                    stl_files.emplace_back(entry.path());
                }
            }
        }

        std::cout << "Total STL files found: " << stl_files.size() << std::endl;

        if (stl_files.empty()) {
            std::cout << "No STL files found matching the criteria." << std::endl;
            return 0;
        }

        // Initialize thread pool with number of hardware threads
        size_t num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4; // Fallback to 4 threads if unable to detect
        ThreadPool pool(num_threads);
        std::cout << "Using " << num_threads << " threads." << std::endl;

        // Atomic counter for processed files
        std::atomic<int> file_count(0);

        // Enqueue tasks
        for (const auto& filepath : stl_files) {
            pool.enqueue([&output_dir, &file_count, filepath, radius]() {
                try {
                    file_count++;
                    std::cout << "Processing file " << file_count.load() << ": " << filepath.filename().string() << std::endl;

                    // Load mesh
                    Mesh mesh = loadSTL(filepath.string());

                    // Center mesh at origin
                    Eigen::Vector3d center_of_mass = computeCenterOfMass(mesh.vertices);
                    Eigen::MatrixXd centered_vertices = mesh.vertices.rowwise() - center_of_mass.transpose();

                    // Normalize to unit sphere
                    Eigen::MatrixXd sphere_vertices = normalizeToUnitSphere(centered_vertices);

                    // Normalize colors
                    Eigen::MatrixXd normalized_colors = normalizeColors(centered_vertices);

                    // Create unique directory for each STL file
                    std::string file_stem = filepath.stem().string();
                    std::string image_dir = output_dir + "/" + file_stem;
                    createDirectory(image_dir);

                    // Compute cylinder projection coordinates
                    Eigen::VectorXd x = sphere_vertices.col(0);
                    Eigen::VectorXd y = sphere_vertices.col(1);
                    Eigen::VectorXd z = sphere_vertices.col(2);

                    Eigen::VectorXd phi(y.size());
                    for (int i = 0; i < y.size(); ++i) {
                        phi(i) = std::atan2(y(i), x(i)); // Use parentheses
                    }
                    Eigen::VectorXd u_cylinder = (phi.array() + M_PI) / (2.0 * M_PI);
                    double z_min = z.minCoeff();
                    double z_max = z.maxCoeff();
                    Eigen::VectorXd z_normalized_cylinder = (z.array() - z_min) / (z_max - z_min);
                    Eigen::VectorXd v_cylinder = z_normalized_cylinder;

                    // Create cylinder projection image (3 channels for color)
                    int width = 1000;
                    int height = 700;
                    cv::Mat cylinder_image = cv::Mat::zeros(height, width, CV_8UC3); // Three channels
                    mapToImage(u_cylinder, v_cylinder, normalized_colors, cylinder_image, cv::Scalar(255, 255, 255), radius);

                    // Save cylinder projection image
                    std::string cylinder_image_path = image_dir + "/cylinder_projection.png";
                    cv::imwrite(cylinder_image_path, cylinder_image);

                    // Compute spherical projection coordinates
                    Eigen::VectorXd norms = sphere_vertices.rowwise().norm();
                    Eigen::VectorXd theta = (z.array() / norms.array()).acos();
                    // Clamp theta to [0, pi] to avoid NaNs
                    for(int i = 0; i < theta.size(); ++i) {
                        if (std::isnan(theta(i))) { // Check for NaN using std::isnan
                            theta(i) = 0.0;
                        }
                    }
                    Eigen::VectorXd u_spherical = phi.array() / (2.0 * M_PI) + 0.5;
                    Eigen::VectorXd v_spherical = theta / M_PI;

                    // Create spherical projection image (3 channels for color)
                    cv::Mat spherical_image = cv::Mat::zeros(height, width, CV_8UC3); // Three channels
                    mapToImage(u_spherical, v_spherical, normalized_colors, spherical_image, cv::Scalar(255, 255, 255));

                    // Save spherical projection image
                    std::string spherical_image_path = image_dir + "/spherical_projection.png";
                    cv::imwrite(spherical_image_path, spherical_image);

                    Eigen::VectorXd xx = mesh.vertices.col(0);
                    Eigen::VectorXd yy = mesh.vertices.col(1);
                    Eigen::VectorXd zz = mesh.vertices.col(2);

                    saveGrayscaleProjection(xx, yy, zz, image_dir + "/"  + "up_projection.png");
                    saveGrayscaleProjection(xx, zz, yy, image_dir + "/"  + "left_projection.png");
                    saveGrayscaleProjection(yy, zz, xx, image_dir + "/"  + "front_projection.png");
                }
                catch (const std::exception& ex) {
                    std::cerr << "Error processing file " << filepath.filename().string() << ": " << ex.what() << std::endl;
                }
            });
        }

        // Wait for all tasks to complete
        pool.wait_until_empty();

        std::cout << "Processing completed. Total files processed: " << file_count.load() << std::endl;
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }
    return 0;
}
