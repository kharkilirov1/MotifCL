#include <motifcl/ops/fog.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/runtime/backend.hpp>
#include <motifcl/runtime/vulkan_backend.hpp>

namespace motifcl {
namespace {

void validate(const Tensor& value, const Tensor& addressed) {
    MCL_CHECK(value.dtype() == DType::F32 && addressed.dtype() == DType::F32,
              "fog_block_product supports f32 only");
    MCL_CHECK(value.ndim() == 2 && addressed.ndim() == 2,
              "fog_block_product expects [rows,d_model]");
    MCL_CHECK(value.shape() == addressed.shape(),
              "fog_block_product input shape mismatch");
    MCL_CHECK(value.backend_ptr() == addressed.backend_ptr(),
              "fog_block_product inputs must share backend");
    MCL_CHECK((value.shape()[1] % 2) == 0,
              "fog_block_product requires even d_model");
}

std::pair<std::vector<float>, std::vector<float>> backward_host(
    const Tensor& value, const Tensor& addressed, const Tensor& grad_out) {
    validate(value, addressed);
    MCL_CHECK(grad_out.dtype() == DType::F32 && grad_out.shape() == value.shape(),
              "fog_block_product backward grad shape mismatch");
    const auto v = value.to_vector<float>();
    const auto r = addressed.to_vector<float>();
    const auto g = grad_out.to_vector<float>();
    const auto rows = static_cast<std::size_t>(value.shape()[0]);
    const auto d = static_cast<std::size_t>(value.shape()[1]);
    std::vector<float> gv(v.size(), 0.0f), gr(v.size(), 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * d;
        std::vector<float> z(d, 0.0f);
        float ss = 0.0f;
        float dot = 0.0f;
        for (std::size_t j = 0; j < d; j += 2) {
            const float vr = v[base + j], vi = v[base + j + 1];
            const float rr = r[base + j], ri = r[base + j + 1];
            const float zr = vr * rr - vi * ri;
            const float zi = vr * ri + vi * rr;
            z[j] = zr; z[j + 1] = zi;
            ss += zr * zr + zi * zi;
            dot += zr * g[base + j] + zi * g[base + j + 1];
        }
        ss = std::max(ss, 1.0e-20f);
        const float scale = std::sqrt(static_cast<float>(d)) / std::sqrt(ss);
        const float corr = dot / ss;
        for (std::size_t j = 0; j < d; j += 2) {
            const float vr = v[base + j], vi = v[base + j + 1];
            const float rr = r[base + j], ri = r[base + j + 1];
            const float gzr = scale * (g[base + j] - z[j] * corr);
            const float gzi = scale * (g[base + j + 1] - z[j + 1] * corr);
            gv[base + j] = gzr * rr + gzi * ri;
            gv[base + j + 1] = -gzr * ri + gzi * rr;
            gr[base + j] = gzr * vr + gzi * vi;
            gr[base + j + 1] = -gzr * vi + gzi * vr;
        }
    }
    return {std::move(gv), std::move(gr)};
}

struct FogBlockProductBackwardNode : autograd::Node {
    Tensor value;
    Tensor addressed;
    FogBlockProductBackwardNode(Tensor v, Tensor r)
        : value(std::move(v)), addressed(std::move(r)) {}
    std::vector<Tensor> inputs() const override { return {value, addressed}; }
    void backward(const Tensor& grad_output) override {
        if (value.requires_grad()) value.backward(fog_block_product_backward_value(value, addressed, grad_output));
        if (addressed.requires_grad()) addressed.backward(fog_block_product_backward_addressed(value, addressed, grad_output));
    }
 };

std::array<std::vector<float>, 7> route_candidate_grads_host(
    const Tensor& logits, const std::array<Tensor,7>&, const Tensor& grad_out) {
    const auto rows=static_cast<std::size_t>(logits.shape()[0]);
    const auto d=static_cast<std::size_t>(grad_out.shape()[1]);
    const auto lv=logits.to_vector<float>();
    const auto gv=grad_out.to_vector<float>();
    std::array<std::vector<float>,7> out;
    for(auto& v:out)v.assign(rows*d,0.0f);
    for(std::size_t r=0;r<rows;++r){
        std::size_t best=0; float bv=lv[r*7];
        for(std::size_t k=1;k<7;++k) if(lv[r*7+k]>bv){bv=lv[r*7+k];best=k;}
        std::copy_n(gv.data()+r*d,d,out[best].data()+r*d);
    }
    return out;
}

std::vector<float> route_logits_grad_host(const Tensor& logits,
                                          const std::array<Tensor,7>& cands,
                                          const Tensor& grad_out) {
    const auto rows=static_cast<std::size_t>(logits.shape()[0]);
    const auto d=static_cast<std::size_t>(grad_out.shape()[1]);
    const auto lv=logits.to_vector<float>(); const auto gv=grad_out.to_vector<float>();
    std::array<std::vector<float>,7> cv; for(int k=0;k<7;++k)cv[k]=cands[k].to_vector<float>();
    std::vector<float> dl(rows*7,0.0f);
    for(std::size_t r=0;r<rows;++r){
        float mx=lv[r*7]; for(int k=1;k<7;++k)mx=std::max(mx,lv[r*7+k]);
        float z=0.0f; float p[7]; for(int k=0;k<7;++k){p[k]=std::exp(lv[r*7+k]-mx);z+=p[k];p[k]/=1.0f;}
        for(int k=0;k<7;++k)p[k]/=z;
        float gp[7]={};
        for(int k=0;k<7;++k) for(std::size_t j=0;j<d;++j) gp[k]+=gv[r*d+j]*cv[k][r*d+j];
        float mean=0.0f; for(int k=0;k<7;++k)mean+=p[k]*gp[k];
        for(int k=0;k<7;++k)dl[r*7+k]=p[k]*(gp[k]-mean);
    }
    return dl;
}

struct FogHardRoute7BackwardNode : autograd::Node {
    Tensor logits; std::array<Tensor,7> candidates;
    FogHardRoute7BackwardNode(Tensor l,std::array<Tensor,7> c):logits(std::move(l)),candidates(std::move(c)){}
    std::vector<Tensor> inputs() const override {
        std::vector<Tensor> in; in.reserve(8); in.push_back(logits); for(const auto& c:candidates)in.push_back(c); return in;
    }
    void backward(const Tensor& grad_output) override {
        const auto rows=static_cast<std::size_t>(logits.shape()[0]);
        const auto d=static_cast<std::size_t>(grad_output.shape()[1]);
        if(logits.requires_grad()){
            Tensor dl;
            if(logits.backend().is_vulkan()){
                dl=Tensor::empty(logits.backend(),logits.shape(),DType::F32);
                std::vector<const VulkanBuffer*> bufs; for(const auto& c:candidates)bufs.push_back(&c.storage().vulkan_buffer);
                auto res=run_vulkan_fog_hard_route7_logits_backward(logits.backend().vulkan_runtime(),logits.storage().vulkan_buffer,bufs,grad_output.storage().vulkan_buffer,dl.storage().vulkan_buffer,rows,d);
                MCL_CHECK(res.success,std::string("FOG route7 logits backward failed: ")+res.error);
            }else{
                auto host=route_logits_grad_host(logits,candidates,grad_output);
                dl=Tensor::from_cpu(logits.backend(),logits.shape(),DType::F32,host.data());
            }
            logits.backward(dl);
        }
        std::array<std::vector<float>,7> host_grads;
        bool host_ready=false;
        for(std::uint32_t k=0;k<7;++k){
            if(!candidates[k].requires_grad())continue;
            Tensor dc;
            if(logits.backend().is_vulkan()){
                dc=Tensor::empty(logits.backend(),candidates[k].shape(),DType::F32);
                auto res=run_vulkan_fog_hard_route7_candidate_backward(logits.backend().vulkan_runtime(),logits.storage().vulkan_buffer,grad_output.storage().vulkan_buffer,dc.storage().vulkan_buffer,rows,d,k);
                MCL_CHECK(res.success,std::string("FOG route7 candidate backward failed: ")+res.error);
            }else{
                if(!host_ready){host_grads=route_candidate_grads_host(logits,candidates,grad_output);host_ready=true;}
                dc=Tensor::from_cpu(logits.backend(),candidates[k].shape(),DType::F32,host_grads[k].data());
            }
            candidates[k].backward(dc);
        }
    }
};

} // namespace

Tensor fog_block_product(const Tensor& value, const Tensor& addressed) {
    validate(value, addressed);
    Tensor out;
    if (value.backend().is_vulkan()) {
        out = Tensor::empty(value.backend(), value.shape(), DType::F32);
        const auto result = run_vulkan_fog_block_product(
            value.backend().vulkan_runtime(),
            value.storage().vulkan_buffer,
            addressed.storage().vulkan_buffer,
            out.storage().vulkan_buffer,
            static_cast<std::size_t>(value.shape()[0]),
            static_cast<std::size_t>(value.shape()[1]));
        MCL_CHECK(result.success, std::string("vulkan fog_block_product failed: ") + result.error);
        autograd::record_op("fog_block_product_vulkan_f32", {value.id(), addressed.id()}, {out.id()});
    } else {
        const auto v = value.to_vector<float>();
        const auto r = addressed.to_vector<float>();
        const auto rows = static_cast<std::size_t>(value.shape()[0]);
        const auto d = static_cast<std::size_t>(value.shape()[1]);
        std::vector<float> y(v.size(), 0.0f);
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t base = row * d;
            float ss = 0.0f;
            for (std::size_t j = 0; j < d; j += 2) {
                const float vr = v[base + j], vi = v[base + j + 1];
                const float rr = r[base + j], ri = r[base + j + 1];
                const float zr = vr * rr - vi * ri;
                const float zi = vr * ri + vi * rr;
                y[base + j] = zr; y[base + j + 1] = zi;
                ss += zr * zr + zi * zi;
            }
            const float scale = std::sqrt(static_cast<float>(d)) / std::sqrt(std::max(ss, 1.0e-20f));
            for (std::size_t j = 0; j < d; ++j) y[base + j] *= scale;
        }
        out = Tensor::from_cpu(value.backend(), value.shape(), DType::F32, y.data());
        autograd::record_op("fog_block_product_host_f32", {value.id(), addressed.id()}, {out.id()}, false);
    }
    if (autograd::is_enabled() && (value.requires_grad() || addressed.requires_grad())) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<FogBlockProductBackwardNode>(value, addressed));
    }
    return out;
}

Tensor fog_block_product_backward_value(const Tensor& value,
                                        const Tensor& addressed,
                                        const Tensor& grad_out) {
    validate(value, addressed);
    if (value.backend().is_vulkan()) {
        auto gv = Tensor::empty(value.backend(), value.shape(), DType::F32);
        auto gr = Tensor::empty(value.backend(), value.shape(), DType::F32);
        const auto result = run_vulkan_fog_block_product_backward(
            value.backend().vulkan_runtime(), value.storage().vulkan_buffer,
            addressed.storage().vulkan_buffer, grad_out.storage().vulkan_buffer,
            gv.storage().vulkan_buffer, gr.storage().vulkan_buffer,
            static_cast<std::size_t>(value.shape()[0]),
            static_cast<std::size_t>(value.shape()[1]));
        MCL_CHECK(result.success, std::string("vulkan fog_block_product backward failed: ") + result.error);
        return gv;
    }
    auto [gv, gr] = backward_host(value, addressed, grad_out);
    (void)gr;
    return Tensor::from_cpu(value.backend(), value.shape(), DType::F32, gv.data());
}

Tensor fog_block_product_backward_addressed(const Tensor& value,
                                            const Tensor& addressed,
                                            const Tensor& grad_out) {
    validate(value, addressed);
    if (value.backend().is_vulkan()) {
        auto gv = Tensor::empty(value.backend(), value.shape(), DType::F32);
        auto gr = Tensor::empty(value.backend(), value.shape(), DType::F32);
        const auto result = run_vulkan_fog_block_product_backward(
            value.backend().vulkan_runtime(), value.storage().vulkan_buffer,
            addressed.storage().vulkan_buffer, grad_out.storage().vulkan_buffer,
            gv.storage().vulkan_buffer, gr.storage().vulkan_buffer,
            static_cast<std::size_t>(value.shape()[0]),
            static_cast<std::size_t>(value.shape()[1]));
        MCL_CHECK(result.success, std::string("vulkan fog_block_product backward failed: ") + result.error);
        return gr;
    }
    auto [gv, gr] = backward_host(value, addressed, grad_out);
    (void)gv;
    return Tensor::from_cpu(value.backend(), value.shape(), DType::F32, gr.data());
}

Tensor fog_hard_route7(const Tensor& logits, const std::array<Tensor,7>& candidates) {
    MCL_CHECK(logits.dtype()==DType::F32 && logits.ndim()==2 && logits.shape()[1]==7,
              "fog_hard_route7 logits must be [rows,7]");
    const auto rows=logits.shape()[0];
    MCL_CHECK(rows>0,"fog_hard_route7 requires rows>0");
    const auto d=candidates[0].shape()[1];
    for(const auto& c:candidates){
        MCL_CHECK(c.dtype()==DType::F32 && c.ndim()==2 && c.shape()[0]==rows && c.shape()[1]==d,
                  "fog_hard_route7 candidate shape mismatch");
        MCL_CHECK(c.backend_ptr()==logits.backend_ptr(),"fog_hard_route7 backend mismatch");
    }
    Tensor out;
    if(logits.backend().is_vulkan()){
        out=Tensor::empty(logits.backend(),{rows,d},DType::F32);
        std::vector<const VulkanBuffer*> bufs; for(const auto& c:candidates)bufs.push_back(&c.storage().vulkan_buffer);
        auto res=run_vulkan_fog_hard_route7(logits.backend().vulkan_runtime(),logits.storage().vulkan_buffer,bufs,out.storage().vulkan_buffer,static_cast<std::size_t>(rows),static_cast<std::size_t>(d));
        MCL_CHECK(res.success,std::string("FOG route7 failed: ")+res.error);
        autograd::record_op("fog_hard_route7_vulkan_f32",{logits.id()},{out.id()});
    }else{
        const auto lv=logits.to_vector<float>(); std::array<std::vector<float>,7> cv;
        for(int k=0;k<7;++k)cv[k]=candidates[k].to_vector<float>();
        std::vector<float> y(static_cast<std::size_t>(rows*d));
        for(int64_t r=0;r<rows;++r){int best=0;float bv=lv[static_cast<std::size_t>(r*7)];for(int k=1;k<7;++k)if(lv[static_cast<std::size_t>(r*7+k)]>bv){bv=lv[static_cast<std::size_t>(r*7+k)];best=k;}std::copy_n(cv[best].data()+static_cast<std::size_t>(r*d),static_cast<std::size_t>(d),y.data()+static_cast<std::size_t>(r*d));}
        out=Tensor::from_cpu(logits.backend(),{rows,d},DType::F32,y.data());
    }
    bool grad=logits.requires_grad(); for(const auto& c:candidates)grad=grad||c.requires_grad();
    if(autograd::is_enabled()&&grad){out.set_requires_grad(true);out._set_grad_fn(std::make_shared<FogHardRoute7BackwardNode>(logits,candidates));}
    return out;
}

} // namespace motifcl
