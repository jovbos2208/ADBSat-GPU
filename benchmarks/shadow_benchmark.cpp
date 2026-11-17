#include "adbsat/core.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc,char** argv){
  Kokkos::ScopeGuard guard(argc,argv);
  const std::size_t per_layer=argc>1?std::stoull(argv[1]):50000;
  const int reps=argc>2?std::stoi(argv[2]):20;
  const std::size_t width=static_cast<std::size_t>(std::ceil(std::sqrt(double(per_layer))));
  std::vector<adbsat::PanelHost> panels;panels.reserve(2*per_layer);
  auto add_layer=[&](double x,double nx,bool reverse){
    for(std::size_t i=0;i<per_layer;++i){
      const double y=double(i%width),z=double(i/width);
      adbsat::PanelHost p{0.32,{nx,0,0},{x,y,z}};
      p.vertex0={x,y-0.4,z-0.4};
      if(!reverse){p.vertex1={x,y+0.4,z-0.4};p.vertex2={x,y,z+0.4};}
      else {p.vertex1={x,y,z+0.4};p.vertex2={x,y+0.4,z-0.4};}
      p.has_vertices=true;panels.push_back(p);
    }
  };
  add_layer(0.0,1.0,false);  // flow-facing targets
  add_layer(1.0,-1.0,true);  // upstream back-facing blockers
  adbsat::Solver solver(panels);
  adbsat::Parameters p;p.model=adbsat::Model::Newton;p.reference_area=double(per_layer)*0.32;p.enable_shadowing=true;
  for(int i=0;i<5;++i)solver.compute(p,{-1,0,0});
  std::vector<double> times;adbsat::Result result;
  for(int i=0;i<reps;++i){result=solver.compute(p,{-1,0,0});times.push_back(result.kernel_seconds);}
  std::sort(times.begin(),times.end());const double median=times[times.size()/2];
  std::cout<<std::setprecision(9)<<"{\"backend\":\""<<Kokkos::DefaultExecutionSpace::name()
    <<"\",\"panels\":"<<panels.size()<<",\"shadowed_panels\":"<<result.shadowed_panel_count
    <<",\"median_seconds\":"<<median<<",\"million_panels_per_second\":"
    <<double(panels.size())/median/1e6<<",\"C_D\":"<<result.drag<<"}\n";
  return result.shadowed_panel_count==per_layer?0:1;
}
