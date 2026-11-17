#include "adbsat/core.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  const std::size_t n = argc > 1 ? std::stoull(argv[1]) : 1000000;
  const int reps = argc > 2 ? std::stoi(argv[2]) : 30;
  const bool shadow = argc > 3 && std::string(argv[3]) == "shadow";
  std::vector<adbsat::PanelHost> panels; panels.reserve(n);
  constexpr double golden=2.39996322972865332;
  for (std::size_t i=0; i<n; ++i) {
    const double z=1.0-2.0*(i+0.5)/double(n);
    const double phi=golden*double(i), r=std::sqrt(std::max(0.0,1.0-z*z));
    std::array<double,3> normal{r*std::cos(phi),r*std::sin(phi),z};
    panels.push_back({4.0*3.14159265358979323846/double(n),normal,normal});
  }
  adbsat::Parameters p; p.model=adbsat::Model::Sentman;
  p.enable_shadowing=shadow;
  p.reference_area=3.14159265358979323846;
  adbsat::Solver solver(panels);
  for (int i=0;i<5;++i) solver.compute(p,{-1,0,0});
  std::vector<double> times; times.reserve(reps);
  adbsat::Result result;
  for (int i=0;i<reps;++i) { result=solver.compute(p,{-1,0,0}); times.push_back(result.kernel_seconds); }
  std::sort(times.begin(),times.end());
  const double median=times[times.size()/2];
  std::cout << std::setprecision(9)
    << "{\"backend\":\"" << Kokkos::DefaultExecutionSpace::name() << "\","
    << "\"panels\":" << n << ",\"repetitions\":" << reps
    << ",\"median_seconds\":" << median << ",\"min_seconds\":" << times.front()
    << ",\"million_panels_per_second\":" << (double(n)/median/1e6)
    << ",\"shadowing\":" << (shadow?"true":"false")
    << ",\"shadowed_panels\":" << result.shadowed_panel_count
    << ",\"C_D\":" << result.drag << "}\n";
}
