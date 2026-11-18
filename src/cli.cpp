#include "adbsat/core.hpp"
#include "cli_common.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  try {
    std::vector<std::string> pos, opts;
    for (int i = 1; i < argc; ++i) {
      std::string a(argv[i]);
      (a.find('=') != std::string::npos ? opts : pos).push_back(a);
    }
    if (pos.size() < 2) {
      std::cerr << "usage: adbsat-cli PANEL.csv MODEL [fx fy fz] [repetitions] [shadow] [KEY=VALUE ...]\n"
                   "CSV: area,nx,ny,nz,cx,cy,cz[,v0x,v0y,v0z,v1x,v1y,v1z,v2x,v2y,v2z]\n"
                   "KEY: alpha_n sigma_t sigma_n alpha speed_ratio wall_temperature freestream_temperature\n"
                   "     freestream_velocity reference_area reference_length cll_beta cll_gamma cll_delta cll_zeta\n";
      return 2;
    }
    auto panels = read_csv(pos[0]);
    adbsat::Parameters params; params.model = adbsat::parse_model(pos[1]);
    for (const auto& o : opts) apply_option(params, o);
    std::array<double,3> flow{-1.0,0.0,0.0};
    if (pos.size() >= 5) flow = {std::stod(pos[2]),std::stod(pos[3]),std::stod(pos[4])};
    int reps = pos.size() >= 6 ? std::max(1, std::stoi(pos[5])) : 1;
    params.enable_shadowing = pos.size() >= 7 && pos[6] == "shadow";
    adbsat::Solver solver(panels);
    adbsat::Result result;
    for (int i=0; i<reps; ++i) result = solver.compute(params, flow, false);
    std::cout << std::setprecision(17)
      << "{\"backend\":\"" << Kokkos::DefaultExecutionSpace::name() << "\","
      << "\"model\":\"" << adbsat::model_name(params.model) << "\","
      << "\"panels\":" << result.panel_count << ",\"repetitions\":" << reps << ','
      << "\"kernel_seconds\":" << result.kernel_seconds << ','
      << "\"shadowed_panels\":" << result.shadowed_panel_count << ','
      << "\"C_D\":" << result.drag << ",\"C_L\":" << result.lift
      << ",\"C_Y\":" << result.side << ",\"Cf_g\":["
      << result.force_ground[0] << ',' << result.force_ground[1] << ',' << result.force_ground[2]
      << "],\"Cm_g\":[" << result.moment_ground[0] << ',' << result.moment_ground[1] << ','
      << result.moment_ground[2] << "]}\n";
  } catch (const std::exception& e) {
    std::cerr << "adbsat-cli: " << e.what() << '\n'; return 1;
  }
}
