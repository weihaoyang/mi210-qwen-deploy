// SPDX-License-Identifier: MIT
// Synthetic operator-only validation; no model or network. Do not select an in-use GPU.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

constexpr int width = 128, tokens = 256;
using Data = std::vector<float>;
static void require(bool ok, const char * msg) { if (!ok) throw std::runtime_error(msg); }
struct Output { Data attn, state; };

class Op {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor *q, *k, *qkv, *g, *b, *s, *y;
    int qh, vh, stride;
public:
    Op(ggml_backend_t dev, int heads, int values) : backend(dev), qh(heads), vh(values), stride(width*(2*qh+vh)) {
        ctx = ggml_init({16*ggml_tensor_overhead()+ggml_graph_overhead(), nullptr, true});
        require(ctx != nullptr, "context allocation failed");
        q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, qh, tokens, 1);
        k = ggml_dup_tensor(ctx, q);
        qkv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, stride, tokens);
        auto *v = ggml_view_4d(ctx, qkv, width, vh, tokens, 1, width*sizeof(float), stride*sizeof(float),
                              stride*tokens*sizeof(float), 2*qh*width*sizeof(float));
        g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, vh, tokens, 1);
        b = ggml_dup_tensor(ctx, g);
        s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, width, vh, 1);
        y = ggml_gated_delta_net(ctx, q, k, v, g, b, s, 1);
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        require(ggml_backend_supports_op(backend, y), "backend does not support GDN");
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "backend buffer allocation failed");
    }
    ~Op() { ggml_backend_buffer_free(buffer); ggml_free(ctx); }
    static void set(ggml_tensor * tensor, const Data & data) {
        require(ggml_nbytes(tensor) == data.size()*sizeof(float), "fixture byte size mismatch");
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    }
    Output run(int seed, const Data & state) {
        Data qd(width*qh*tokens), kd(qd.size()), vd(stride*tokens, 123.0f), gd(vh*tokens), bd(gd.size());
        for (int t=0;t<tokens;++t) {
            for (int h=0;h<qh;++h) for (int i=0;i<width;++i) {
                const size_t j=(t*qh+h)*width+i;
                qd[j]=0.05f*std::sin(0.07f*(i+11*h+3*t+seed));
                kd[j]=0.05f*std::cos(0.11f*(i+7*h+2*t+seed));
            }
            for (int h=0;h<vh;++h) {
                gd[t*vh+h]=-0.03f - ((t+h+seed)%7)*0.01f;
                bd[t*vh+h]=0.2f + ((t+3*h+seed)%11)*0.04f;
                for (int i=0;i<width;++i)
                    vd[t*stride+(2*qh+h)*width+i]=0.35f*std::sin(0.031f*(i+3*h+2*t+9*seed));
            }
        }
        set(q,qd); set(k,kd); set(qkv,vd); set(g,gd); set(b,bd); set(s,state);
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS, "GDN compute failed");
        Output out{Data(width*vh*tokens), Data(width*width*vh)};
        ggml_backend_tensor_get(y,out.attn.data(),0,out.attn.size()*sizeof(float));
        ggml_backend_tensor_get(y,out.state.data(),out.attn.size()*sizeof(float),out.state.size()*sizeof(float));
        return out;
    }
};

double nmse(const Data & ref, const Data & got) {
    require(ref.size()==got.size(), "shape mismatch");
    double error=0.0, energy=0.0;
    for (size_t i=0;i<ref.size();++i) {
        require(std::isfinite(got[i]), "non-finite device result");
        const double d=double(ref[i])-got[i];
        error+=d*d; energy+=double(ref[i])*ref[i];
    }
    return error/std::max(energy,1e-30);
}

int main(int argc, char ** argv) try {
    if (argc != 4) throw std::runtime_error("usage: gfx90a-gdn-device BACKEND Q_HEADS V_HEADS");
    const int qh=std::stoi(argv[2]), vh=std::stoi(argv[3]);
    require(qh>0 && vh<=64 && vh>=qh && vh%qh==0, "head shape invalid");
    ggml_backend_load_all();
    auto * cpu=ggml_backend_init_by_name("CPU",nullptr);
    auto * other=ggml_backend_init_by_name(argv[1],nullptr);
    require(cpu && other, "configured backend not available");
    {
        Op ref(cpu,qh,vh), candidate(other,qh,vh);
        const Data zero(width*width*vh,0.0f);
        const auto a=ref.run(1,zero), ax=candidate.run(1,zero);
        const auto b=ref.run(7,a.state), bx=candidate.run(7,ax.state);
        const auto r=ref.run(7,zero), rx=candidate.run(7,zero);
        for (const auto & [title, expected, got] : {
                std::tuple{"fresh", &a, &ax}, std::tuple{"continued", &b, &bx}, std::tuple{"reset", &r, &rx}}) {
            const double output=nmse(expected->attn,got->attn), state=nmse(expected->state,got->state);
            std::cout<<title<<" output_nmse="<<output<<" state_nmse="<<state<<'\n';
            require(output<=2e-7 && state<=2e-7, "FP32-reference NMSE gate failed");
        }
        require(nmse(r.attn,b.attn)>1e-7,"input state control is not discriminating");
    }
    ggml_backend_free(other); ggml_backend_free(cpu);
    return 0;
} catch (const std::exception & e) {
    std::cerr<<e.what()<<'\n';
    return 1;
}
