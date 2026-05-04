#include <cmath>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "example_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        std::vector<float> a = {1, 2, 3, 4, 5, 6};      // 2x3
        std::vector<float> b = {7, 8, 9, 10, 11, 12};   // 3x2
        auto A = motifcl::Tensor::from_cpu(backend, {2, 3}, motifcl::DType::F32, a.data());
        auto B = motifcl::Tensor::from_cpu(backend, {3, 2}, motifcl::DType::F32, b.data());
        auto C = motifcl::matmul(A, B);
        auto out = C.to_vector<float>();
        std::cout << "C =";
        for (auto v : out) std::cout << ' ' << v;
        std::cout << "\nexpected: 58 64 139 154\n";
    } catch (const std::exception& e) {
        return motifcl_example::handle_exception(e, "02_matmul");
    }
}
