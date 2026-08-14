#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include "example_utils.hpp"

namespace {
constexpr float kPi = 3.14159265358979323846f;

int arg_int(char** argv, int argc, int idx, int fallback) {
    if (idx >= argc) return fallback;
    return std::max(1, std::atoi(argv[idx]));
}
float arg_float(char** argv, int argc, int idx, float fallback) {
    if (idx >= argc) return fallback;
    return std::max(1e-8f, std::strtof(argv[idx], nullptr));
}

std::vector<float> state_code(int state, int n_states, int d_model) {
    std::vector<float> out(static_cast<std::size_t>(d_model));
    const int pairs=d_model/2;
    const float amp=std::sqrt(2.0f);
    for(int j=0;j<pairs;++j){
        const int h=j+1;
        const float angle=2.0f*kPi*static_cast<float>(state*h)/static_cast<float>(n_states);
        out[static_cast<std::size_t>(2*j)]=amp*std::cos(angle);
        out[static_cast<std::size_t>(2*j+1)]=amp*std::sin(angle);
    }
    return out;
}

void install_state_codebook(motifcl::nn::FogV3Model& model, int n_states) {
    auto host=model.lexical.token_embedding.weight.data.to_vector<float>();
    const int vocab=model.config.vocab_size;
    const int d=model.config.d_model;
    if(vocab < 4+n_states) throw std::runtime_error("gate vocab too small");
    for(int s=0;s<n_states;++s){
        const auto code=state_code(s,n_states,d);
        std::copy(code.begin(),code.end(),host.begin()+static_cast<std::ptrdiff_t>((4+s)*d));
    }
    auto& backend=model.lexical.token_embedding.weight.data.backend();
    model.lexical.token_embedding.weight.data=motifcl::Tensor::from_cpu(backend,{vocab,d},motifcl::DType::F32,host.data());
    model.lexical.token_embedding.weight.data.set_requires_grad(false);
    model.lexical.token_embedding.weight.trainable=false;
}

struct Batch {
    std::vector<std::int32_t> query;
    std::vector<std::int32_t> keys;
    std::vector<std::int32_t> values;
    std::vector<std::int32_t> target;
};

Batch make_batch(int n_states,int batch,int depth,std::mt19937& rng){
    std::uniform_int_distribution<int> state_dist(0,n_states-1);
    Batch out;
    out.query.resize(static_cast<std::size_t>(batch));
    out.keys.resize(static_cast<std::size_t>(batch*n_states));
    out.values.resize(static_cast<std::size_t>(batch*n_states));
    out.target.resize(static_cast<std::size_t>(batch));
    std::vector<int> table(static_cast<std::size_t>(batch*n_states));
    for(int b=0;b<batch;++b){
        for(int s=0;s<n_states;++s){
            const int operand=state_dist(rng);
            table[static_cast<std::size_t>(b*n_states+s)]=operand;
            out.keys[static_cast<std::size_t>(b*n_states+s)]=4+s;
            out.values[static_cast<std::size_t>(b*n_states+s)]=4+operand;
        }
        int cur=state_dist(rng);
        out.query[static_cast<std::size_t>(b)]=4+cur;
        for(int t=0;t<depth;++t){
            const int operand=table[static_cast<std::size_t>(b*n_states+cur)];
            cur=(cur+operand)%n_states;
        }
        out.target[static_cast<std::size_t>(b)]=4+cur;
    }
    return out;
}

int argmax_row(const float* p,int n){int best=0;for(int i=1;i<n;++i)if(p[i]>p[best])best=i;return best;}

struct Eval { float accuracy=0.f; float block_fraction=0.f; };

Eval evaluate(motifcl::nn::FogV3Model& model,motifcl::Backend& backend,int n_states,int depth,int examples,std::uint32_t seed){
    std::mt19937 rng(seed);
    int correct=0,total=0,block=0,routes=0;
    const int vocab=model.config.vocab_size;
    while(total<examples){
        const int bs=std::min(64,examples-total);
        auto bh=make_batch(n_states,bs,depth,rng);
        auto q=motifcl::Tensor::from_cpu(backend,{bs},motifcl::DType::I32,bh.query.data());
        auto k=motifcl::Tensor::from_cpu(backend,{bs,n_states},motifcl::DType::I32,bh.keys.data());
        auto v=motifcl::Tensor::from_cpu(backend,{bs,n_states},motifcl::DType::I32,bh.values.data());
        auto state=model.initial_state(q);
        for(int t=0;t<depth;++t){
            auto step=model.structured_step(state,k,v,bs,n_states);
            const auto lh=step.operator_logits.to_vector<float>();
            for(int b=0;b<bs;++b){ if(argmax_row(lh.data()+static_cast<std::size_t>(b*7),7)==2) ++block; ++routes; }
            state=step.state;
        }
        const auto logits=model.direct_vocab_logits(state.value).to_vector<float>();
        for(int b=0;b<bs;++b){
            const int pred=argmax_row(logits.data()+static_cast<std::size_t>(b*vocab),vocab);
            if(pred==bh.target[static_cast<std::size_t>(b)]) ++correct;
        }
        total+=bs;
    }
    return {static_cast<float>(correct)/static_cast<float>(total),static_cast<float>(block)/static_cast<float>(std::max(routes,1))};
}

} // namespace

int main(int argc,char** argv){
    using namespace motifcl;
    try{
        const int steps=arg_int(argv,argc,1,300);
        const int batch=arg_int(argv,argc,2,64);
        const float lr=arg_float(argv,argc,3,2e-3f);
        const int n_states=arg_int(argv,argc,4,8);
        const auto probe=probe_vulkan_runtime();
        if(!probe.available()){std::cerr<<"No Vulkan compute device: "<<probe.error<<"\n";return 77;}
        auto backend=Backend::create_vulkan();
        manual_seed(20260815u);
        std::mt19937 rng(20261815u);

        nn::FogV3Config cfg;
        cfg.vocab_size=32; cfg.max_seq_len=64; cfg.d_model=32; cfg.n_heads=4;
        cfg.n_layers=1; cfg.d_ff=64; cfg.dropout=0.0f;
        nn::FogV3Model model(backend,cfg);
        install_state_codebook(model,n_states);
        for(auto* p:model.binder.parameters()){ if(p){ p->trainable=false; p->data.set_requires_grad(false); } }
        auto train_params=model.machine.parameters();
        optim::Adam opt(train_params,lr,0.9f,0.95f,1e-8f,1e-4f);

        std::cout<<"FOG v3 structured recurrent Vulkan gate\n"
                 <<"device="<<backend.device_info().device_name
                 <<" steps="<<steps<<" batch="<<batch<<" states="<<n_states<<" lr="<<lr<<"\n";
        float first_loss=0.f,last_loss=0.f;
        std::uniform_int_distribution<int> depth_dist(1,3);
        for(int step_i=1;step_i<=steps;++step_i){
            const int depth=depth_dist(rng);
            auto bh=make_batch(n_states,batch,depth,rng);
            auto q=Tensor::from_cpu(backend,{batch},DType::I32,bh.query.data());
            auto k=Tensor::from_cpu(backend,{batch,n_states},DType::I32,bh.keys.data());
            auto v=Tensor::from_cpu(backend,{batch,n_states},DType::I32,bh.values.data());
            auto target=Tensor::from_cpu(backend,{batch},DType::I32,bh.target.data());
            auto state=model.initial_state(q);
            Tensor last_route;
            for(int t=0;t<depth;++t){ auto rs=model.structured_step(state,k,v,batch,n_states); state=rs.state; last_route=rs.operator_logits; }
            auto logits=model.direct_vocab_logits(state.value);
            auto loss=softmax_cross_entropy(logits,target);
            loss.backward(); opt.step(); opt.zero_grad(); backend.finish();
            const float lv=loss.item(); if(step_i==1)first_loss=lv; last_loss=lv;
            if(step_i==1 || step_i%25==0 || step_i==steps){
                const auto lh=last_route.to_vector<float>(); int block=0;
                for(int b=0;b<batch;++b) if(argmax_row(lh.data()+static_cast<std::size_t>(b*7),7)==2) ++block;
                std::cout<<"step="<<step_i<<"/"<<steps<<" depth="<<depth<<" loss="<<std::setprecision(7)<<lv<<" block_route="<<block<<"/"<<batch<<"\n";
            }
        }
        bool pass=true;
        for(int depth: {1,2,3,4,6,8}){
            const auto ev=evaluate(model,backend,n_states,depth,512,20270000u+static_cast<std::uint32_t>(depth));
            std::cout<<"eval depth="<<depth<<" accuracy="<<std::setprecision(6)<<ev.accuracy<<" block_route="<<ev.block_fraction<<"\n";
            if(ev.accuracy<0.995f)pass=false;
        }
        std::cout<<"structured_gate_summary first_loss="<<first_loss<<" last_loss="<<last_loss<<"\n";
        if(!pass){std::cerr<<"FOG structured recurrent gate FAILED\n";return 2;}
        std::cout<<"FOG structured recurrent gate PASSED\n";
        return 0;
    }catch(const std::exception& e){return motifcl_example::handle_exception(e,"11_fog_v3_structured_machine_gate");}
}
