#pragma once

#include <string>
#include <vector>

#include <motifcl/nn/parameter.hpp>
#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

void save_tensor(const Tensor& tensor, const std::string& path);
Tensor load_tensor(Backend& backend, const std::string& path);

void save_parameters(const std::vector<nn::Parameter*>& params, const std::string& path);
void load_parameters(const std::vector<nn::Parameter*>& params, Backend& backend, const std::string& path);

} // namespace motifcl
