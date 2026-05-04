#include <cmath>
#include <vector>

#include <motifcl/motifcl.hpp>

#include "test_utils.hpp"

namespace {

motifcl::Tensor make_tensor_after_backend_scope() {
    auto backend = motifcl::Backend::create_opencl();
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
    return motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::F32, values.data());
}

} // namespace

int main() {
    try {
        auto tensor = make_tensor_after_backend_scope();
        auto host = tensor.to_vector<float>();
        if (host.size() != 4) return 1;
        for (int i = 0; i < 4; ++i) {
            if (std::fabs(host[static_cast<std::size_t>(i)] - static_cast<float>(i + 1)) > 1e-6f) return 2;
        }
        bool threw = false;
        try {
            (void)tensor.backend();
        } catch (const std::exception&) {
            threw = true;
        }
        if (!threw) return 3;
        return 0;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
