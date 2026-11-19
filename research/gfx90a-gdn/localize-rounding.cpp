// SPDX-License-Identifier: MIT
// CPU-only single-row FP64 emulation of selective binary16 operand casts.
// This does not simulate GPU ISA reduction order, denorm handling, or all heads.
#include <array>
#include <cmath>
#include <cstdio>
#include <utility>
constexpr int W=128, T=256, C=16;
using Row=std::array<double,W>;
using Seq=std::array<Row,T>;
using Scalars=std::array<double,T>;
enum {QK=1, INTER=2, INTRA=4, UPDATE=8, PRED=16};
bool flush_half_denormals=false;
constexpr int RESCALE=32, EXP=256;
double narrow(double x,bool enable) {
    if (!enable) return x;
    const double h=double(static_cast<_Float16>(x));
    return flush_half_denormals && std::abs(h)<0x1p-14 ? 0.0 : h;
}
double dot(const Row&a,const Row&b,double scale=1,bool half=false) {
    double r=0;
    for(int w=0;w<W;++w) r+=narrow(a[w]*scale,half)*narrow(b[w],half);
    return r;
}
struct Fixture {
    Seq q{},k{}; Scalars v{},g{},b{};
    explicit Fixture(int caseid,int h=0,int col=0) {
        const int kh=h%8;
        for(int t=0;t<T;++t) {
            for(int w=0;w<W;++w) {
                q[t][w]=float(0.05f*std::sin(0.07f*(w+11*kh+3*t+caseid)));
                k[t][w]=float(0.05f*std::cos(0.11f*(w+7*kh+2*t+caseid)));
            }
            v[t]=float(0.35f*std::sin(0.031f*(col+3*h+2*t+9*caseid)));
            g[t]=float(-0.03f-((t+h+caseid)%7)*0.01f);
            b[t]=float(0.2f+((t+3*h+caseid)%11)*0.04f);
        }
    }
};
using Result=std::pair<Scalars,Row>;

Result recurrent(const Fixture& f,Row state) {
    Scalars y{};
    const double scale=1/std::sqrt(double(W));
    for(int t=0;t<T;++t) {
        const double decay=std::exp(f.g[t]);
        const double delta=f.b[t]*(f.v[t]-decay*dot(f.k[t],state));
        for(int w=0;w<W;++w) state[w]=decay*state[w]+delta*f.k[t][w];
        y[t]=dot(f.q[t],state,scale);
    }
    return {y,state};
}

Result run(const Fixture& f,Row state,int mode) {
    Scalars y{};
    const double scale=1/std::sqrt(double(W));
    for(int base=0;base<T;base+=C) {
        std::array<double,C> p{},u{},z{};
        std::array<Row,C> kcd{};
        for(int i=0;i<C;++i) {
            const int t=base+i;
            p[i]=(i?p[i-1]:0)+f.g[t];
            u[i]=f.b[t]*f.v[t];
            for(int w=0;w<W;++w) kcd[i][w]=f.b[t]*std::exp(p[i])*f.k[t][w];
            for(int j=0;j<i;++j) {
                const double l=f.b[t]*std::exp(p[i]-p[j])*dot(f.k[t],f.k[base+j]);
                u[i]-=l*u[j];
                for(int w=0;w<W;++w) kcd[i][w]-=l*kcd[j][w];
            }
            z[i]=u[i]-dot(kcd[i],state,1,mode&PRED);
            double local=0;
            for(int j=0;j<=i;++j) {
                const double mul=(mode & RESCALE) ? EXP : 1;
                const double qk=dot(f.q[t],f.k[base+j],scale*mul,mode&QK)*std::exp(p[i]-p[j])/mul;
                local+=(narrow(qk*mul,mode&INTRA)/mul)*narrow(z[j],mode&INTRA);
            }
            const double mul=(mode & RESCALE) ? EXP : 1;
            y[t]=local+(dot(f.q[t],state,scale*mul,mode&INTER)/mul)*std::exp(p[i]);
        }
        for(int w=0;w<W;++w) {
            state[w]*=std::exp(p.back());
            for(int j=0;j<C;++j) state[w]+=narrow(f.k[base+j][w]*std::exp(p.back()-p[j]),mode&UPDATE)*narrow(z[j],mode&UPDATE);
        }
    }
    return {y,state};
}
template<class A> double nmse(const A&a,const A&b) {
    double e=0,s=0;
    for(size_t i=0;i<a.size();++i) { e+=(a[i]-b[i])*(a[i]-b[i]); s+=a[i]*a[i]; }
    return e/s;
}
int main() {
    constexpr std::array<int,6> heads={0,1,7,8,16,23};
    constexpr std::array<int,4> rows={0,31,32,127};
    constexpr std::array<int,12> masks={QK,INTER,INTRA,UPDATE,PRED,31,31^QK,31^INTER,31^INTRA,31^UPDATE,31^PRED,31|RESCALE};
    constexpr std::array<const char*,12> names={"qk","inter","intra","update","prediction","all",
                "all_except_qk","all_except_inter","all_except_intra","all_except_update","all_except_prediction","all_rescaled"};
    for(bool ftz : {false,true}) for(int seed : {1,7}) {
        flush_half_denormals=ftz;
        std::array<double,12> out_squared{}, state_squared{}, max_out{};
        double out_norm=0.0,state_norm=0.0;
        for(int h:heads) for(int col:rows) {
            const Fixture f(seed,h,col);
            const auto exact=recurrent(f,{});
            const auto check=run(f,{},0);
            if(nmse(exact.first,check.first)>1e-24 || nmse(exact.second,check.second)>1e-24) return 1;
            double yo=0.0,so=0.0;
            for(auto e:exact.first) yo+=e*e;
            for(auto e:exact.second) so+=e*e;
            out_norm+=yo;
            state_norm+=so;
            for(size_t i=0;i<masks.size();++i) {
                const auto res=run(f,{},masks[i]);
                const auto err=nmse(exact.first,res.first);
                out_squared[i]+=err*yo;
                state_squared[i]+=nmse(exact.second,res.second)*so;
                if(err>max_out[i]) max_out[i]=err;
            }
        }
        for(size_t i=0;i<masks.size();++i)
            std::printf("ftz=%d seed=%d stage=%-22s output_nmse=%.8g state_nmse=%.8g max_row_nmse=%.8g\n",int(ftz),seed,names[i],
                        out_squared[i]/out_norm,state_squared[i]/state_norm,max_out[i]);
    }
}
