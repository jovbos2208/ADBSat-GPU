#include "adbsat/core.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace adbsat {
namespace {
constexpr double pi = 3.1415926535897932384626433832795;

struct Accumulator {
  double fx, fy, fz, mx, my, mz;
  unsigned long long shadowed;
  KOKKOS_INLINE_FUNCTION Accumulator() : fx(0), fy(0), fz(0), mx(0), my(0), mz(0), shadowed(0) {}
  KOKKOS_INLINE_FUNCTION void operator+=(const Accumulator& rhs) {
    fx += rhs.fx; fy += rhs.fy; fz += rhs.fz;
    mx += rhs.mx; my += rhs.my; mz += rhs.mz;
    shadowed += rhs.shadowed;
  }
};

KOKKOS_INLINE_FUNCTION
bool ray_box(const double ox, const double oy, const double oz,
             const double dx, const double dy, const double dz,
             const Kokkos::View<double*[3]>& bmin,
             const Kokkos::View<double*[3]>& bmax, const int node,
             const double eps) {
  double tmin = eps, tmax = 1.0e300;
  const double origin[3] = {ox, oy, oz};
  const double dir[3] = {dx, dy, dz};
  for (int axis=0; axis<3; ++axis) {
    if (fabs(dir[axis]) < 1.0e-15) {
      if (origin[axis] < bmin(node,axis) || origin[axis] > bmax(node,axis)) return false;
    } else {
      const double inv = 1.0 / dir[axis];
      double a = (bmin(node,axis)-origin[axis])*inv;
      double b = (bmax(node,axis)-origin[axis])*inv;
      if (a>b) { const double tmp=a; a=b; b=tmp; }
      tmin=fmax(tmin,a); tmax=fmin(tmax,b);
      if (tmax<tmin) return false;
    }
  }
  return true;
}

KOKKOS_INLINE_FUNCTION
bool ray_triangle(const double ox, const double oy, const double oz,
                  const double dx, const double dy, const double dz,
                  const Kokkos::View<double*[3]>& v0,
                  const Kokkos::View<double*[3]>& v1,
                  const Kokkos::View<double*[3]>& v2, const int tri,
                  const double eps) {
  const double e1x=v1(tri,0)-v0(tri,0), e1y=v1(tri,1)-v0(tri,1), e1z=v1(tri,2)-v0(tri,2);
  const double e2x=v2(tri,0)-v0(tri,0), e2y=v2(tri,1)-v0(tri,1), e2z=v2(tri,2)-v0(tri,2);
  const double px=dy*e2z-dz*e2y, py=dz*e2x-dx*e2z, pz=dx*e2y-dy*e2x;
  const double det=e1x*px+e1y*py+e1z*pz;
  if (fabs(det)<eps) return false;
  const double inv=1.0/det;
  const double tx=ox-v0(tri,0), ty=oy-v0(tri,1), tz=oz-v0(tri,2);
  const double u=(tx*px+ty*py+tz*pz)*inv;
  if (u<0.0 || u>1.0) return false;
  const double qx=ty*e1z-tz*e1y, qy=tz*e1x-tx*e1z, qz=tx*e1y-ty*e1x;
  const double v=(dx*qx+dy*qy+dz*qz)*inv;
  if (v<0.0 || u+v>=1.0) return false; // match the Python/MATLAB open third edge
  const double t=(e2x*qx+e2y*qy+e2z*qz)*inv;
  return t>eps;
}

KOKKOS_INLINE_FUNCTION
void local_coefficients(const Model model, const Parameters& p, const double c, const double sn,
                        double& cp, double& ctau) {
  const bool back = c < 0.0;
  cp = 0.0;
  ctau = 0.0;
  if (model == Model::Newton) {
    if (!back) cp = 2.0 * c * c;
    return;
  }
  if (model == Model::Cook) {
    const double tinf = p.freestream_velocity * p.freestream_velocity /
                        (3.0 * p.mean_specific_gas_constant);
    const double root = sqrt(fmax(0.0, 1.0 + p.alpha * (p.wall_temperature / tinf - 1.0)));
    double cd = 2.0 * c * (1.0 + (2.0 / 3.0) * root * c);
    double cl = (4.0 / 3.0) * root * sn * c;
    if (back) cd = cl = 0.0;
    cp = cd * c + cl * sn;
    ctau = cd * sn - cl * c;
    return;
  }
  if (model == Model::StorchHyp) {
    const double vw = sqrt(pi * p.mean_specific_gas_constant * p.wall_temperature / 2.0);
    if (!back) {
      cp = 2.0 * c * (p.sigma_n * vw / p.freestream_velocity + (2.0 - p.sigma_n) * c);
      ctau = 2.0 * c * sn * p.sigma_t;
    }
    return;
  }

  const double s = p.speed_ratio;
  const double e = exp(-s * s * c * c);
  const double erf_term = erf(s * c);
  if (model == Model::Schaaf) {
    const double tr = sqrt(p.wall_temperature / p.freestream_temperature);
    cp = (((2.0 - p.sigma_n) * s / sqrt(pi) * c + 0.5 * p.sigma_n * tr) * e +
          ((2.0 - p.sigma_n) * (0.5 + s * s * c * c) +
           0.5 * p.sigma_n * tr * sqrt(pi) * s * c) * (1.0 + erf_term)) / (s * s);
    ctau = p.sigma_t * sn / (s * sqrt(pi)) *
            (e + s * sqrt(pi) * c * (1.0 + erf_term));
    return;
  }
  if (model == Model::Sentman) {
    const double ti = 0.5 * s * s * p.freestream_temperature;
    const double thermal = sqrt(fmax(0.0, 0.5 * (1.0 + p.alpha * (p.wall_temperature / ti - 1.0))));
    cp = (c * c + 1.0 / (2.0 * s * s)) * (1.0 + erf_term) +
         c / (sqrt(pi) * s) * e + 0.5 * thermal *
         (sqrt(pi) * c * (1.0 + erf_term) + e / s);
    ctau = sn * c * (1.0 + erf_term) + sn / (s * sqrt(pi)) * e;
    return;
  }
  if (model == Model::Maxwell || model == Model::PICLasMaxwell) {
    const double accommodation = model == Model::Maxwell ? p.alpha : p.momentum_accommodation;
    const double spec = 1.0 - accommodation;
    const double tref = model == Model::Maxwell
      ? p.wall_temperature / p.freestream_temperature
      : fmax(0.0, 1.0 - p.trans_accommodation +
                  p.trans_accommodation * p.wall_temperature / p.freestream_temperature);
    // cos(2*(pi/2-delta)) = 1 - 2*cos(delta)^2.  Avoid acos/trigonometry.
    const double cos2theta = 1.0 - 2.0 * c * c;
    double cd = 2.0 * (1.0 - spec * cos2theta) / (sqrt(pi) * s) * e +
      c / (s * s) * (1.0 + 2.0 * s * s + spec * (1.0 - 2.0 * s * s * cos2theta)) * erf_term +
      accommodation / s * sqrt(pi) * c * c * sqrt(tref);
    double cl = 4.0 * spec / (sqrt(pi) * s) * c * sn * e +
      sn / (s * s) * (1.0 + spec * (1.0 + 4.0 * s * s * c * c)) * erf_term +
      accommodation / s * sqrt(pi) * c * sn * sqrt(tref);
    if (back) cd = cl = 0.0;
    cp = cd * c + cl * sn;
    ctau = cd * sn - cl * c;
    return;
  }
  // DRIA scalar-mixture approximation. Full species-specific DRIA belongs in a future extension.
  const double g = 1.0 / (2.0 * s * s);
  const double z = 1.0 + erf_term;
  const double vr = sqrt(fmax(0.0, 0.5 * (1.0 + p.alpha *
      (4.0 * p.mean_specific_gas_constant * p.wall_temperature /
       (p.freestream_velocity * p.freestream_velocity) - 1.0))));
  const double pp = e / s;
  cp = pp / sqrt(pi) + c * (1.0 + g) * z + 0.5 * c * vr * (c * sqrt(pi) * z + pp);
  ctau = sn * g * z + 0.5 * sn * vr * (c * sqrt(pi) * z + pp);
}
}  // namespace

Model parse_model(const std::string& name) {
  if (name == "Newton") return Model::Newton;
  if (name == "Cook") return Model::Cook;
  if (name == "Maxwell") return Model::Maxwell;
  if (name == "PICLasMaxwell") return Model::PICLasMaxwell;
  if (name == "Schaaf") return Model::Schaaf;
  if (name == "Sentman") return Model::Sentman;
  if (name == "StorchHyp") return Model::StorchHyp;
  if (name == "DRIA") return Model::DRIA;
  throw std::invalid_argument("unsupported GSI model: " + name);
}

const char* model_name(Model model) {
  switch (model) {
    case Model::Newton: return "Newton"; case Model::Cook: return "Cook";
    case Model::Maxwell: return "Maxwell"; case Model::PICLasMaxwell: return "PICLasMaxwell";
    case Model::Schaaf: return "Schaaf"; case Model::Sentman: return "Sentman";
    case Model::StorchHyp: return "StorchHyp"; case Model::DRIA: return "DRIA";
  }
  return "Unknown";
}

Solver::Solver(const std::vector<PanelHost>& panels)
    : panel_count_(panels.size()), normals_("normals", panels.size()),
      centers_("centers", panels.size()), areas_("areas", panels.size()),
      cp_("cp", panels.size()), ctau_("ctau", panels.size()) {
  if (panels.empty()) throw std::invalid_argument("mesh must contain at least one panel");
  auto nh = Kokkos::create_mirror_view(normals_);
  auto ch = Kokkos::create_mirror_view(centers_);
  auto ah = Kokkos::create_mirror_view(areas_);
  struct Triangle { std::array<double,3> a,b,c,centroid,bmin,bmax; int panel; };
  std::vector<Triangle> triangles; triangles.reserve(panels.size());
  for (std::size_t i = 0; i < panels.size(); ++i) {
    if (!(panels[i].area > 0.0)) throw std::invalid_argument("panel area must be positive");
    ah(i) = panels[i].area;
    double norm = 0.0;
    for (int j = 0; j < 3; ++j) norm += panels[i].normal[j] * panels[i].normal[j];
    norm = std::sqrt(norm);
    if (!(norm > 0.0)) throw std::invalid_argument("panel normal must be nonzero");
    for (int j = 0; j < 3; ++j) {
      nh(i, j) = panels[i].normal[j] / norm;
      ch(i, j) = panels[i].center[j];
    }
    Triangle tri{}; tri.panel=static_cast<int>(i);
    if (panels[i].has_vertices) {
      tri.a=panels[i].vertex0; tri.b=panels[i].vertex1; tri.c=panels[i].vertex2;
    } else {
      // Area-preserving equilateral proxy. Exact shadowing requires supplied vertices.
      const std::array<double,3> n{nh(i,0),nh(i,1),nh(i,2)};
      const std::array<double,3> seed = fabs(n[0])<0.9 ? std::array<double,3>{1,0,0}
                                                       : std::array<double,3>{0,1,0};
      std::array<double,3> u{n[1]*seed[2]-n[2]*seed[1], n[2]*seed[0]-n[0]*seed[2], n[0]*seed[1]-n[1]*seed[0]};
      const double un=std::sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
      for (double& value:u) value/=un;
      const std::array<double,3> v{n[1]*u[2]-n[2]*u[1], n[2]*u[0]-n[0]*u[2], n[0]*u[1]-n[1]*u[0]};
      const double radius=std::sqrt(4.0*panels[i].area/std::sqrt(3.0))/std::sqrt(3.0);
      for (int k=0;k<3;++k) {
        tri.a[k]=panels[i].center[k]+radius*u[k];
        tri.b[k]=panels[i].center[k]+radius*(-0.5*u[k]+0.5*std::sqrt(3.0)*v[k]);
        tri.c[k]=panels[i].center[k]+radius*(-0.5*u[k]-0.5*std::sqrt(3.0)*v[k]);
      }
    }
    for (int k=0;k<3;++k) {
      tri.centroid[k]=(tri.a[k]+tri.b[k]+tri.c[k])/3.0;
      tri.bmin[k]=std::min({tri.a[k],tri.b[k],tri.c[k]});
      tri.bmax[k]=std::max({tri.a[k],tri.b[k],tri.c[k]});
    }
    triangles.push_back(tri);
  }
  Kokkos::deep_copy(normals_, nh); Kokkos::deep_copy(centers_, ch); Kokkos::deep_copy(areas_, ah);

  struct HostNode { std::array<double,3> bmin,bmax; int left=-1,right=-1,start=0,count=0; };
  std::vector<int> order(triangles.size()); std::iota(order.begin(),order.end(),0);
  std::vector<HostNode> nodes; nodes.reserve(2*triangles.size());
  std::function<int(int,int)> build=[&](int begin,int end) {
    const int id=static_cast<int>(nodes.size()); nodes.push_back({});
    HostNode node;
    for (int k=0;k<3;++k) { node.bmin[k]=std::numeric_limits<double>::infinity(); node.bmax[k]=-node.bmin[k]; }
    for (int pos=begin;pos<end;++pos) for(int k=0;k<3;++k) {
      node.bmin[k]=std::min(node.bmin[k],triangles[order[pos]].bmin[k]);
      node.bmax[k]=std::max(node.bmax[k],triangles[order[pos]].bmax[k]);
    }
    if (end-begin<=4) { node.start=begin; node.count=end-begin; }
    else {
      int axis=0;
      if (node.bmax[1]-node.bmin[1]>node.bmax[axis]-node.bmin[axis]) axis=1;
      if (node.bmax[2]-node.bmin[2]>node.bmax[axis]-node.bmin[axis]) axis=2;
      const int mid=(begin+end)/2;
      std::nth_element(order.begin()+begin,order.begin()+mid,order.begin()+end,
        [&](int a,int b){return triangles[a].centroid[axis]<triangles[b].centroid[axis];});
      node.left=build(begin,mid); node.right=build(mid,end);
    }
    nodes[id]=node; return id;
  };
  build(0,static_cast<int>(triangles.size()));

  tri_v0_=Kokkos::View<double*[3]>("triangle_v0",panels.size());
  tri_v1_=Kokkos::View<double*[3]>("triangle_v1",panels.size());
  tri_v2_=Kokkos::View<double*[3]>("triangle_v2",panels.size());
  triangle_panel_=Kokkos::View<int*>("triangle_panel",panels.size());
  bvh_min_=Kokkos::View<double*[3]>("bvh_min",nodes.size()); bvh_max_=Kokkos::View<double*[3]>("bvh_max",nodes.size());
  bvh_left_=Kokkos::View<int*>("bvh_left",nodes.size()); bvh_right_=Kokkos::View<int*>("bvh_right",nodes.size());
  bvh_start_=Kokkos::View<int*>("bvh_start",nodes.size()); bvh_count_=Kokkos::View<int*>("bvh_count",nodes.size());
  auto v0h=Kokkos::create_mirror_view(tri_v0_), v1h=Kokkos::create_mirror_view(tri_v1_), v2h=Kokkos::create_mirror_view(tri_v2_);
  auto tph=Kokkos::create_mirror_view(triangle_panel_);
  for(std::size_t pos=0;pos<order.size();++pos) { const auto& t=triangles[order[pos]]; tph(pos)=t.panel;
    for(int k=0;k<3;++k){v0h(pos,k)=t.a[k];v1h(pos,k)=t.b[k];v2h(pos,k)=t.c[k];}}
  auto bminh=Kokkos::create_mirror_view(bvh_min_), bmaxh=Kokkos::create_mirror_view(bvh_max_);
  auto lefth=Kokkos::create_mirror_view(bvh_left_), righth=Kokkos::create_mirror_view(bvh_right_);
  auto starth=Kokkos::create_mirror_view(bvh_start_), counth=Kokkos::create_mirror_view(bvh_count_);
  for(std::size_t i=0;i<nodes.size();++i){lefth(i)=nodes[i].left;righth(i)=nodes[i].right;starth(i)=nodes[i].start;counth(i)=nodes[i].count;
    for(int k=0;k<3;++k){bminh(i,k)=nodes[i].bmin[k];bmaxh(i,k)=nodes[i].bmax[k];}}
  Kokkos::deep_copy(tri_v0_,v0h);Kokkos::deep_copy(tri_v1_,v1h);Kokkos::deep_copy(tri_v2_,v2h);Kokkos::deep_copy(triangle_panel_,tph);
  Kokkos::deep_copy(bvh_min_,bminh);Kokkos::deep_copy(bvh_max_,bmaxh);Kokkos::deep_copy(bvh_left_,lefth);Kokkos::deep_copy(bvh_right_,righth);
  Kokkos::deep_copy(bvh_start_,starth);Kokkos::deep_copy(bvh_count_,counth);
}

Result Solver::compute(const Parameters& p, const std::array<double, 3>& flow, bool copy_coeffs) {
  if (!(p.reference_area > 0.0) || !(p.reference_length > 0.0) || !(p.speed_ratio > 0.0))
    throw std::invalid_argument("reference dimensions and speed ratio must be positive");
  double flow_norm = std::sqrt(flow[0]*flow[0] + flow[1]*flow[1] + flow[2]*flow[2]);
  if (!(flow_norm > 0.0)) throw std::invalid_argument("flow direction must be nonzero");
  const double ux = flow[0] / flow_norm, uy = flow[1] / flow_norm, uz = flow[2] / flow_norm;
  auto normals = normals_; auto centers = centers_; auto areas = areas_; auto cpv = cp_; auto ctv = ctau_;
  auto tv0=tri_v0_; auto tv1=tri_v1_; auto tv2=tri_v2_; auto tpanel=triangle_panel_;
  auto bmin=bvh_min_; auto bmax=bvh_max_; auto bleft=bvh_left_; auto bright=bvh_right_;
  auto bstart=bvh_start_; auto bcount=bvh_count_;
  Accumulator sums;
  Kokkos::Timer timer;
  Kokkos::parallel_reduce("adbsat_panel_kernel", Kokkos::RangePolicy<>(0, panel_count_),
    KOKKOS_LAMBDA(const int i, Accumulator& acc) {
      const double nx = normals(i,0), ny = normals(i,1), nz = normals(i,2);
      double cosine = -(ux*nx + uy*ny + uz*nz);
      cosine = fmax(-1.0, fmin(1.0, cosine));
      const double sine = sqrt(fmax(0.0, 1.0 - cosine*cosine));
      double cp, ct; local_coefficients(p.model, p, cosine, sine, cp, ct);
      bool shadowed=false;
      if (p.enable_shadowing && cosine>=-1.7453292519943296e-6) { // Python's 90.0001 degree split
        const double dx=-ux, dy=-uy, dz=-uz;
        int stack[64]; int top=0; stack[top++]=0;
        while(top>0 && !shadowed) {
          const int node=stack[--top];
          if(!ray_box(centers(i,0),centers(i,1),centers(i,2),dx,dy,dz,bmin,bmax,node,p.shadow_epsilon)) continue;
          if(bcount(node)>0) {
            for(int q=0;q<bcount(node) && !shadowed;++q) {
              const int tri=bstart(node)+q, other=tpanel(tri);
              if(other==i) continue;
              // Python only admits back-facing panels as occluders.
              if(ux*normals(other,0)+uy*normals(other,1)+uz*normals(other,2)<=1.7453292519943296e-6) continue;
              shadowed=ray_triangle(centers(i,0),centers(i,1),centers(i,2),dx,dy,dz,tv0,tv1,tv2,tri,p.shadow_epsilon);
            }
          } else {
            if(top+2<=64){stack[top++]=bleft(node);stack[top++]=bright(node);}
          }
        }
      }
      if(shadowed){cp=0.0;ct=0.0;acc.shadowed+=1;}
      if (copy_coeffs) { cpv(i) = cp; ctv(i) = ct; }
      double tx = ux - (ux*nx + uy*ny + uz*nz)*nx;
      double ty = uy - (ux*nx + uy*ny + uz*nz)*ny;
      double tz = uz - (ux*nx + uy*ny + uz*nz)*nz;
      const double tn = sqrt(tx*tx + ty*ty + tz*tz);
      if (tn > 0.0) { tx /= tn; ty /= tn; tz /= tn; } else { tx=ty=tz=0.0; }
      const double scale = areas(i) / p.reference_area;
      const double fx = scale * (ct*tx - cp*nx);
      const double fy = scale * (ct*ty - cp*ny);
      const double fz = scale * (ct*tz - cp*nz);
      acc.fx += fx; acc.fy += fy; acc.fz += fz;
      acc.mx += (centers(i,1)*fz - centers(i,2)*fy) / p.reference_length;
      acc.my += (centers(i,2)*fx - centers(i,0)*fz) / p.reference_length;
      acc.mz += (centers(i,0)*fy - centers(i,1)*fx) / p.reference_length;
    }, sums);
  Kokkos::fence();
  const double elapsed = timer.seconds();
  if (copy_coeffs) {
    auto cph = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cp_);
    auto cth = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ctau_);
    cp_host_.resize(panel_count_); ctau_host_.resize(panel_count_);
    for (std::size_t i=0; i<panel_count_; ++i) { cp_host_[i]=cph(i); ctau_host_[i]=cth(i); }
  }
  Result r;
  r.force_ground = {sums.fx, sums.fy, sums.fz};
  r.moment_ground = {sums.mx, sums.my, sums.mz};
  // Flow-aligned drag; lift/side use global z/y to preserve the Python zero-angle convention.
  // Positive drag is aligned with the incident flow direction.
  r.drag = sums.fx*ux + sums.fy*uy + sums.fz*uz;
  r.lift = sums.fz; r.side = sums.fy;
  r.kernel_seconds = elapsed; r.panel_count = panel_count_;
  r.shadowed_panel_count = static_cast<std::size_t>(sums.shadowed);
  return r;
}
}  // namespace adbsat
