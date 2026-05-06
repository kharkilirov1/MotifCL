#pragma once

#include <string>
#include <vector>

#include <motifcl/nn/parameter.hpp>

namespace motifcl::train {

struct GradientClipResult {
    float total_norm = 0.0f;
    float clip_coef = 1.0f;
    bool clipped = false;
};

struct TrainingHistory {
    std::vector<float> losses;
    std::vector<float> learning_rates;
};

GradientClipResult clip_grad_norm(const std::vector<nn::Parameter*>& params, float max_norm, float eps = 1e-6f);
void write_history_csv(const TrainingHistory& history, const std::string& path);

} // namespace motifcl::train
