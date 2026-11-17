#include "adbsat/core.hpp"
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
      adbsat::Model::StorchHyp, adbsat::Model::DRIA}) {
    p.model=model;
    auto r=symmetric_solver.compute(p,{-1,0,0});
    if (!std::isfinite(r.drag)) throw std::runtime_error("non-finite model result");
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
