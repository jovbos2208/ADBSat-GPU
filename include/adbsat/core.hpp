#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace adbsat {

enum class Model { Newton, Cook, Maxwell, PICLasMaxwell, Schaaf, Sentman, StorchHyp, DRIA };

Model parse_model(const std::string& name);
const char* model_name(Model model);

struct Parameters {
  Model model = Model::Sentman;
  double alpha = 0.9;
  double sigma_n = 0.9;
  double sigma_t = 0.7;
  double momentum_accommodation = 0.81;
  double trans_accommodation = 0.9;
  double wall_temperature = 300.0;
  double freestream_temperature = 1000.0;
  double speed_ratio = 7.0;
  double freestream_velocity = 7500.0;
  double mean_specific_gas_constant = 296.8;
  double reference_area = 1.0;
  double reference_length = 1.0;
  bool enable_shadowing = false;
  double shadow_epsilon = 1.0e-7;
};

struct PanelHost {
  double area;
  std::array<double, 3> normal;
  std::array<double, 3> center;
  std::array<double, 3> vertex0{};
  std::array<double, 3> vertex1{};
  std::array<double, 3> vertex2{};
  bool has_vertices = false;
};

struct Result {
  std::array<double, 3> force_ground{};
  std::array<double, 3> moment_ground{};
  double drag = 0.0;
  double lift = 0.0;
  double side = 0.0;
  double kernel_seconds = 0.0;
  std::size_t panel_count = 0;
  std::size_t shadowed_panel_count = 0;
};

class Solver {
 public:
  explicit Solver(const std::vector<PanelHost>& panels);
  Result compute(const Parameters& params, const std::array<double, 3>& flow_direction,
                 bool copy_panel_coefficients = false);
  const std::vector<double>& pressure_coefficients() const { return cp_host_; }
  const std::vector<double>& shear_coefficients() const { return ctau_host_; }
  std::size_t size() const { return panel_count_; }

 private:
  std::size_t panel_count_{};
  Kokkos::View<double*[3]> normals_;
  Kokkos::View<double*[3]> centers_;
  Kokkos::View<double*> areas_;
  Kokkos::View<double*> cp_;
  Kokkos::View<double*> ctau_;
  Kokkos::View<double*[3]> tri_v0_;
  Kokkos::View<double*[3]> tri_v1_;
  Kokkos::View<double*[3]> tri_v2_;
  Kokkos::View<double*[3]> bvh_min_;
  Kokkos::View<double*[3]> bvh_max_;
  Kokkos::View<int*> bvh_left_;
  Kokkos::View<int*> bvh_right_;
  Kokkos::View<int*> bvh_start_;
  Kokkos::View<int*> bvh_count_;
  Kokkos::View<int*> triangle_panel_;
  std::vector<double> cp_host_;
  std::vector<double> ctau_host_;
};

}  // namespace adbsat
