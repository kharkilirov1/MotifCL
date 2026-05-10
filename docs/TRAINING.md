# Training guide

MotifCL now has the pieces needed for repeatable small training runs:

- `manual_seed(seed)` for deterministic host RNG-backed tensor factories/dropout masks.
- `train::TensorDataLoader` for cyclic row mini-batches.
- `train::StepLR`, `train::CosineLR`, and `train::ConstantLR`.
- `train::clip_grad_norm`.
- `train::TrainingHistory` and `write_history_csv`.
- `save_parameters` / `load_parameters` checkpoints.
- `optim::DynamicLossScaler` and `optim::MixedPrecisionAdam` for FP32-master/loss-scaling experiments.

Minimal C++ pattern:

```cpp
motifcl::manual_seed(2026);
auto backend = motifcl::Backend::create_opencl();
motifcl::nn::Sequential model({...});
motifcl::optim::Adam opt(model.parameters(), 2e-2f);
motifcl::train::TensorDataLoader loader(x, y, 32);
motifcl::train::CosineLR schedule(2e-2f, 1000, 1e-4f);
motifcl::train::Trainer trainer(model, opt, [&] { return loader.next(); }, motifcl::mse_loss);
auto history = trainer.fit_with_history(1000, 50, 1.0f, &schedule);
motifcl::save_parameters(model.parameters(), "checkpoint.mclp");
motifcl::train::write_history_csv(history, "history.csv");
```

See `examples/cpp/07_long_train_checkpoint.cpp` for a longer OpenCL training example.

Mixed-precision infrastructure pattern:

```cpp
motifcl::optim::MixedPrecisionAdam opt(model.parameters(), 1e-3f);
motifcl::optim::DynamicLossScaler scaler(65536.0f);

auto loss = criterion(model.forward(batch_x), batch_y);
auto scaled = scaler.scale_loss(loss);
scaled.backward();
if (!opt.step_scaled(scaler)) {
    // overflow: optimizer step skipped and loss scale backed off
}
opt.zero_grad();
```

Boundary: this provides FP32 master weights, unscale/overflow checks, and F16/F32 parameter sync. It does not magically add FP16 backward kernels for every operator; unsupported FP16 autograd paths still need explicit kernels.
