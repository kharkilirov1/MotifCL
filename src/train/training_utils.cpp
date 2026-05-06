#include <motifcl/train/training_utils.hpp>

#include <motifcl/core/error.hpp>
#include <motifcl/ops/basic_ops.hpp>

#include <cmath>
#include <fstream>

namespace motifcl::train {

GradientClipResult clip_grad_norm(const std::vector<nn::Parameter*>& params, float max_norm, float eps) {
    MCL_CHECK(max_norm > 0.0f, "max_norm must be positive");
    double sum_sq = 0.0;
    for (auto* p : params) {
        if (!p || !p->trainable || !p->grad()) continue;
        MCL_CHECK(p->grad()->dtype() == DType::F32, "clip_grad_norm supports f32 gradients only");
        const auto values = p->grad()->to_vector<float>();
        for (float v : values) sum_sq += static_cast<double>(v) * static_cast<double>(v);
    }
    GradientClipResult result;
    result.total_norm = static_cast<float>(std::sqrt(sum_sq));
    result.clip_coef = max_norm / (result.total_norm + eps);
    result.clipped = result.clip_coef < 1.0f;
    if (result.clipped) {
        for (auto* p : params) {
            if (!p || !p->trainable || !p->grad()) continue;
            scale_inplace(*p->grad(), result.clip_coef);
        }
    } else {
        result.clip_coef = 1.0f;
    }
    return result;
}

void write_history_csv(const TrainingHistory& history, const std::string& path) {
    std::ofstream out(path);
    MCL_CHECK(out.good(), "failed to open training history CSV: " + path);
    out << "step,loss,lr\n";
    for (std::size_t i = 0; i < history.losses.size(); ++i) {
        const float lr = i < history.learning_rates.size() ? history.learning_rates[i] : 0.0f;
        out << (i + 1) << ',' << history.losses[i] << ',' << lr << '\n';
    }
}

} // namespace motifcl::train
