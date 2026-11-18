#include "adbsat/core.hpp"
#include <algorithm>
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

// Optional KEY=VALUE arguments (any position after MODEL) set model parameters.
static void apply_option(adbsat::Parameters& p, const std::string& arg) {
  const auto eq = arg.find('=');
  const std::string key = arg.substr(0, eq);
  const double v = std::stod(arg.substr(eq + 1));
  if (key == "alpha_n") p.alpha_n = v;
  else if (key == "sigma_t") p.sigma_t = v;
  else if (key == "sigma_n") p.sigma_n = v;
  else if (key == "alpha") p.alpha = v;
  else if (key == "speed_ratio") p.speed_ratio = v;
  else if (key == "wall_temperature") p.wall_temperature = v;
  else if (key == "freestream_temperature") p.freestream_temperature = v;
  else if (key == "freestream_velocity") p.freestream_velocity = v;
  else if (key == "reference_area") p.reference_area = v;
  else if (key == "reference_length") p.reference_length = v;
  else if (key == "cll_beta") p.cll_beta = v;
  else if (key == "cll_gamma") p.cll_gamma = v;
  else if (key == "cll_delta") p.cll_delta = v;
  else if (key == "cll_zeta") p.cll_zeta = v;
  else throw std::invalid_argument("unknown option: " + key);
}

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
