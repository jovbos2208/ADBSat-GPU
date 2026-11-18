#pragma once
// Shared helpers for the adbsat command-line tools (adbsat-cli, adbsat-batch).
#include "adbsat/core.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

inline std::vector<adbsat::PanelHost> read_csv(const std::string& path) {
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
inline void apply_option(adbsat::Parameters& p, const std::string& arg) {
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

