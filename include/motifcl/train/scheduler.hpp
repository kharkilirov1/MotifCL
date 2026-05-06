#pragma once

#include <algorithm>
#include <cmath>

namespace motifcl::train {

class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual float lr_at(int step) const = 0;
};

class ConstantLR : public LRScheduler {
public:
    explicit ConstantLR(float lr) : lr_(lr) {}
    float lr_at(int) const override { return lr_; }

private:
    float lr_ = 1e-3f;
};

class StepLR : public LRScheduler {
public:
    StepLR(float initial_lr, int step_size, float gamma)
        : initial_lr_(initial_lr), step_size_(step_size), gamma_(gamma) {}

    float lr_at(int step) const override {
        const int k = step_size_ > 0 ? std::max(0, step - 1) / step_size_ : 0;
        return initial_lr_ * std::pow(gamma_, static_cast<float>(k));
    }

private:
    float initial_lr_ = 1e-3f;
    int step_size_ = 1;
    float gamma_ = 0.1f;
};

class CosineLR : public LRScheduler {
public:
    CosineLR(float initial_lr, int total_steps, float min_lr = 0.0f)
        : initial_lr_(initial_lr), total_steps_(total_steps), min_lr_(min_lr) {}

    float lr_at(int step) const override {
        constexpr float pi = 3.14159265358979323846f;
        const float t = total_steps_ > 1
            ? std::min(1.0f, std::max(0.0f, static_cast<float>(step - 1) / static_cast<float>(total_steps_ - 1)))
            : 1.0f;
        return min_lr_ + 0.5f * (initial_lr_ - min_lr_) * (1.0f + std::cos(pi * t));
    }

private:
    float initial_lr_ = 1e-3f;
    int total_steps_ = 1;
    float min_lr_ = 0.0f;
};

} // namespace motifcl::train
