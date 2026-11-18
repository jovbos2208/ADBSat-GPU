#include "adbsat/core.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

static void close(double actual, double expected, double tol, const char* what) {
  if (std::abs(actual-expected) > tol) throw std::runtime_error(std::string(what)+" mismatch");
}

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  std::vector<adbsat::PanelHost> panels{{1.0, {1,0,0}, {0,0,0}}};
  adbsat::Solver solver(panels);
  adbsat::Parameters p; p.model=adbsat::Model::Newton; p.reference_area=1.0;
  auto front=solver.compute(p, {-1,0,0}, true);
  close(front.drag, 2.0, 1e-13, "front-face drag");
  close(solver.pressure_coefficients().at(0), 2.0, 1e-13, "front-face cp");
  auto back=solver.compute(p, {1,0,0});
  close(back.drag, 0.0, 1e-13, "back-face drag");

  std::vector<adbsat::PanelHost> symmetric{
    {1.0,{1,0,0},{0,1,0}}, {1.0,{1,0,0},{0,-1,0}}};
  adbsat::Solver symmetric_solver(symmetric);
  p.reference_area=2.0;
  auto result=symmetric_solver.compute(p,{-1,0,0});
  close(result.drag,2.0,1e-13,"symmetric drag");
  close(result.moment_ground[2],0.0,1e-13,"symmetric moment");

  for (auto model : {adbsat::Model::Cook, adbsat::Model::Maxwell,
      adbsat::Model::PICLasMaxwell, adbsat::Model::Schaaf, adbsat::Model::Sentman,
      adbsat::Model::StorchHyp, adbsat::Model::DRIA, adbsat::Model::CLL}) {
    p.model=model;
    auto r=symmetric_solver.compute(p,{-1,0,0});
    if (!std::isfinite(r.drag)) throw std::runtime_error("non-finite model result");
  }

  // CLL (Walker closed form): single-panel drag against ADBSat-PyVersion
  // calc/coeff_CLL.py (atomic oxygen, V=7718.371 m/s, T_inf=961 K, T_w=300 K).
  // Columns: angle between normal and upstream direction [deg], alpha_n, sigma_t, C_D.
  const double cll_ref[][4] = {
    {0, 0.29999999999999999, 0.69999999999999996, 3.7053819187323587},
    {40, 0.94999999999999996, 0.69999999999999996, 1.5658833795507177},
    {80, 0.5, 0.20000000000000001, 0.090624453216811007},
    {120, 0.29999999999999999, 0.90000000000000002, 4.8352863066571349e-10},
    {30, 1, 0.69999999999999996, 1.7128395555929365},          // alpha_n = 1 branch
    {0, 0.99999899999999997, 1, 2.2222888299717876}};
  adbsat::Parameters q; q.model=adbsat::Model::CLL; q.reference_area=1.0;
  q.speed_ratio=7.7229550927843027; q.wall_temperature=300.0; q.freestream_temperature=961.0;
  for (const auto& c : cll_ref) {
    const double d=c[0]*3.1415926535897932384626433832795/180.0;
    adbsat::Solver one({{1.0,{std::cos(d),std::sin(d),0.0},{0,0,0}}});
    q.alpha_n=c[1]; q.sigma_t=c[2];
    const double cd=one.compute(q,{-1,0,0}).drag;
    if (std::abs(cd-c[3]) > 1e-12*std::max(1.0,std::abs(c[3])) + 1e-18)
      throw std::runtime_error("CLL single-panel drag mismatch");
  }

  adbsat::PanelHost target{0.5,{1,0,0},{0,0,0}};
  target.vertex0={0,-1,-1}; target.vertex1={0,1,-1}; target.vertex2={0,0,1}; target.has_vertices=true;
  adbsat::PanelHost blocker{0.5,{-1,0,0},{1,0,0}};
  blocker.vertex0={1,-1,-1}; blocker.vertex1={1,0,1}; blocker.vertex2={1,1,-1}; blocker.has_vertices=true;
  adbsat::Solver shadow_solver({target,blocker});
  p.model=adbsat::Model::Newton; p.reference_area=0.5; p.enable_shadowing=true;
  auto shadowed=shadow_solver.compute(p,{-1,0,0});
  close(shadowed.drag,0.0,1e-13,"shadowed drag");
  if(shadowed.shadowed_panel_count!=1) throw std::runtime_error("shadow count mismatch");
  std::cout << "PASS backend=" << Kokkos::DefaultExecutionSpace::name() << '\n';
}
