#include "adbsat/core.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

static std::vector<adbsat::PanelHost> read_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open mesh CSV: " + path);
  std::vector<adbsat::PanelHost> panels;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream row(line);
    adbsat::PanelHost p{};
    if (!(row >> p.area >> p.normal[0] >> p.normal[1] >> p.normal[2]
              >> p.center[0] >> p.center[1] >> p.center[2])) {
      if (panels.empty()) continue;  // optional header
      throw std::runtime_error("invalid CSV row: " + line);
    }
    if (row >> p.vertex0[0] >> p.vertex0[1] >> p.vertex0[2]
            >> p.vertex1[0] >> p.vertex1[1] >> p.vertex1[2]
            >> p.vertex2[0] >> p.vertex2[1] >> p.vertex2[2]) p.has_vertices=true;
    panels.push_back(p);
  }
  return panels;
}

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  try {
    if (argc < 3) {
      std::cerr << "usage: adbsat-cli PANEL.csv MODEL [fx fy fz] [repetitions] [shadow]\n"
                   "CSV: area,nx,ny,nz,cx,cy,cz[,v0x,v0y,v0z,v1x,v1y,v1z,v2x,v2y,v2z]\n";
      return 2;
    }
    auto panels = read_csv(argv[1]);
    adbsat::Parameters params; params.model = adbsat::parse_model(argv[2]);
    std::array<double,3> flow{-1.0,0.0,0.0};
    if (argc >= 6) flow = {std::stod(argv[3]),std::stod(argv[4]),std::stod(argv[5])};
    int reps = argc >= 7 ? std::max(1, std::stoi(argv[6])) : 1;
    params.enable_shadowing = argc >= 8 && std::string(argv[7]) == "shadow";
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
