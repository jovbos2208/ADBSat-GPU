// adbsat-batch: evaluate one panel geometry for many states with the geometry
// kept on the device (one Solver, repeated compute() calls).
//
//   adbsat-batch PANEL.csv MODEL STATES.csv [shadow] [KEY=VALUE ...]
//
// STATES.csv: one state per line, columns fx,fy,fz[,alpha_n,sigma_t]
// (flow direction in ground axes; optional per-state CLL parameters; a
// non-numeric first line is treated as a header). KEY=VALUE sets the common
// model parameters as in adbsat-cli. Output on stdout, one CSV line per state:
// C_D,C_L,C_Y,Cfx,Cfy,Cfz,Cmx,Cmy,Cmz,shadowed_panels
// (moments about the coordinate origin, normalized by reference_area*reference_length).
#include "cli_common.hpp"
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  try {
    std::vector<std::string> pos, opts;
    for (int i = 1; i < argc; ++i) {
      std::string a(argv[i]);
      (a.find('=') != std::string::npos ? opts : pos).push_back(a);
    }
    if (pos.size() < 3) {
      std::cerr << "usage: adbsat-batch PANEL.csv MODEL STATES.csv [shadow] [KEY=VALUE ...]\n"
                   "STATES.csv: fx,fy,fz[,alpha_n,sigma_t] per line\n";
      return 2;
    }
    auto panels = read_csv(pos[0]);
    adbsat::Parameters params; params.model = adbsat::parse_model(pos[1]);
    for (const auto& o : opts) apply_option(params, o);
    params.enable_shadowing = pos.size() >= 4 && pos[3] == "shadow";
    std::ifstream in(pos[2]);
    if (!in) throw std::runtime_error("cannot open states CSV: " + pos[2]);
    adbsat::Solver solver(panels);
    std::cout << std::setprecision(17) << "C_D,C_L,C_Y,Cfx,Cfy,Cfz,Cmx,Cmy,Cmz,shadowed_panels\n";
    std::string line; bool first = true;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream row(line);
      std::array<double, 3> flow{};
      if (!(row >> flow[0] >> flow[1] >> flow[2])) {
        if (first) { first = false; continue; }  // header
        throw std::runtime_error("invalid state row: " + line);
      }
      first = false;
      adbsat::Parameters p = params;
      double an, st;
      if (row >> an >> st) { p.alpha_n = an; p.sigma_t = st; }
      const auto r = solver.compute(p, flow, false);
      std::cout << r.drag << ',' << r.lift << ',' << r.side << ',' << r.force_ground[0] << ',' << r.force_ground[1]
                << ',' << r.force_ground[2] << ',' << r.moment_ground[0] << ',' << r.moment_ground[1] << ','
                << r.moment_ground[2] << ',' << r.shadowed_panel_count << '\n';
    }
  } catch (const std::exception& e) {
    std::cerr << "adbsat-batch: " << e.what() << '\n'; return 1;
  }
}
