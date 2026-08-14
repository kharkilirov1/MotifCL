#include <cmath>
#include <iostream>
#include <vector>

#include <motifcl/motifcl.hpp>

namespace {
bool close(float a, float b, float tol = 2e-4f) { return std::fabs(a - b) <= tol; }
}

int main() {
    using namespace motifcl;
    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::cerr << "Skipping FOG Vulkan ops test: " << probe.error << "\n";
        return 77;
    }
    try {
        auto backend = Backend::create_vulkan();
        const std::vector<float> av = {1.f, -2.f, 0.5f, 3.f};
        const std::vector<float> bv = {2.f, 4.f, -1.f, 0.25f};
        auto a = Tensor::from_cpu(backend, {1,4}, DType::F32, av.data());
        auto b = Tensor::from_cpu(backend, {1,4}, DType::F32, bv.data());
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        auto m = mul(a,b);
        const auto mh = m.to_vector<float>();
        const float mref[4] = {2.f,-8.f,-0.5f,0.75f};
        for (int i=0;i<4;++i) if (!close(mh[static_cast<std::size_t>(i)],mref[i])) return 1;

        auto si = silu(a);
        const auto sh = si.to_vector<float>();
        for (int i=0;i<4;++i) {
            const float x=av[static_cast<std::size_t>(i)];
            const float ref=x/(1.f+std::exp(-x));
            if (!close(sh[static_cast<std::size_t>(i)],ref)) return 1;
        }
        auto sg = sigmoid(a);
        const auto sgh=sg.to_vector<float>();
        for (int i=0;i<4;++i) {
            const float ref=1.f/(1.f+std::exp(-av[static_cast<std::size_t>(i)]));
            if (!close(sgh[static_cast<std::size_t>(i)],ref)) return 1;
        }

        // (1+2i)*(3+4i)=(-5+10i), (0.5-i)*(2+0i)=(1-2i), then normalize to sqrt(4)=2.
        const std::vector<float> vv={1.f,2.f,0.5f,-1.f};
        const std::vector<float> rr={3.f,4.f,2.f,0.f};
        auto v=Tensor::from_cpu(backend,{1,4},DType::F32,vv.data());
        auto r=Tensor::from_cpu(backend,{1,4},DType::F32,rr.data());
        v.set_requires_grad(true); r.set_requires_grad(true);
        auto bp=fog_block_product(v,r);
        const auto bh=bp.to_vector<float>();
        const float norm=std::sqrt(25.f+100.f+1.f+4.f);
        const float sc=2.f/norm;
        const float bref[4]={-5.f*sc,10.f*sc,1.f*sc,-2.f*sc};
        for(int i=0;i<4;++i) if(!close(bh[static_cast<std::size_t>(i)],bref[i],4e-4f)) return 1;

        // Hard 7-way FOG routing: forward must select candidate 2, while
        // straight-through backward must still produce a gradient for logits.
        const std::vector<float> logits_v={-2.f,-1.f,5.f,-3.f,-4.f,-5.f,-6.f};
        auto logits=Tensor::from_cpu(backend,{1,7},DType::F32,logits_v.data());
        logits.set_requires_grad(true);
        std::array<Tensor,7> cands;
        for(int k=0;k<7;++k){
            std::vector<float> cv(4,static_cast<float>(k+1));
            cands[static_cast<std::size_t>(k)]=Tensor::from_cpu(backend,{1,4},DType::F32,cv.data());
            cands[static_cast<std::size_t>(k)].set_requires_grad(true);
        }
        auto routed=fog_hard_route7(logits,cands);
        const auto rh=routed.to_vector<float>();
        for(float x:rh) if(!close(x,3.f)) return 1;

        auto loss=mean_all(add(add(add(m,si),bp),routed));
        loss.backward();
        if(!a.grad() || !b.grad() || !v.grad() || !r.grad() || !logits.grad() || !cands[2].grad()) {
            std::cerr << "missing FOG op gradient\n"; return 1;
        }

        // End-to-end structured v3 smoke: exact query identity addresses one
        // key/value row, READ-biased machine keeps that payload, cosine-tied
        // head should prefer the corresponding value token. This exercises
        // Vulkan embedding -> split-value GQA binder -> hard grammar -> readout
        // and a backward pass through the complete machine slice.
        nn::FogV3Config cfg;
        cfg.vocab_size=64; cfg.max_seq_len=32; cfg.d_model=32; cfg.n_heads=4;
        cfg.n_layers=1; cfg.d_ff=64; cfg.dropout=0.0f;
        nn::FogV3Model small(backend,cfg);
        const std::vector<std::int32_t> qid={5};
        const std::vector<std::int32_t> kids={3,5,7,9};
        const std::vector<std::int32_t> vids={10,11,12,13};
        auto qi=Tensor::from_cpu(backend,{1},DType::I32,qid.data());
        auto ki=Tensor::from_cpu(backend,{1,4},DType::I32,kids.data());
        auto vi=Tensor::from_cpu(backend,{1,4},DType::I32,vids.data());
        auto st=small.initial_state(qi);
        auto step=small.structured_step(st,ki,vi,1,4);
        auto vocab_logits=small.direct_vocab_logits(step.state.value);
        const auto vl=vocab_logits.to_vector<float>();
        int pred=0; for(int i=1;i<64;++i) if(vl[static_cast<std::size_t>(i)]>vl[static_cast<std::size_t>(pred)]) pred=i;
        if(pred!=11){ std::cerr << "FOG structured smoke expected token 11, got " << pred << "\n"; return 1; }
        const std::vector<std::int32_t> target_id={11};
        auto tgt=Tensor::from_cpu(backend,{1},DType::I32,target_id.data());
        auto machine_loss=softmax_cross_entropy(vocab_logits,tgt);
        machine_loss.backward();
        bool machine_grad=false;
        for(auto* p:small.machine_parameters()) if(p && p->grad()){ machine_grad=true; break; }
        if(!machine_grad){ std::cerr << "missing full FOG machine gradient\n"; return 1; }

        std::cout << "FOG Vulkan ops + structured machine OK on " << probe.devices.front().name << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FOG Vulkan ops failed: " << e.what() << "\n";
        return 1;
    }
}
