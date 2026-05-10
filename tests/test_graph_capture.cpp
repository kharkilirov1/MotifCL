#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include <motifcl/motifcl.hpp>

#include "test_utils.hpp"

namespace {

bool same_ids(const std::vector<int>& a, const std::vector<int>& b) {
    return a == b;
}

} // namespace

int main() {
    namespace ag = motifcl::autograd;

    try {
        auto backend = motifcl::Backend::create_opencl();
        std::vector<float> a = {1, 2, 3, 4, 5, 6};
        std::vector<float> b = {7, 8, 9, 10, 11, 12};
        auto A = motifcl::Tensor::from_cpu(backend, {2, 3}, motifcl::DType::F32, a.data());
        auto B = motifcl::Tensor::from_cpu(backend, {3, 2}, motifcl::DType::F32, b.data());

        ag::begin_graph_capture();
        if (!ag::is_graph_capturing()) return 1;

        auto C = motifcl::matmul(A, B);
        if (ag::current_graph_capture().size() != 1) return 2;

        auto D = motifcl::gelu(C);
        auto E = motifcl::add(D, D);
        auto V = E.view({4});

        auto graph = ag::end_graph_capture();
        if (ag::is_graph_capturing()) return 3;
        if (graph.size() != 4) {
            std::cerr << "Expected 4 captured graph nodes, got " << graph.size() << "\n";
            return 4;
        }
        if (!graph.replayable()) return 14;
        if (graph.schedule().size() != graph.size()) return 15;
        auto specs = graph.tensor_specs_list();
        if (specs.size() < 6) return 19;
        bool saw_c_spec = false;
        bool saw_dynamic_spec = false;
        for (const auto& spec : specs) {
            if (spec.id == C.id()) {
                saw_c_spec = spec.shape == C.shape() && spec.dtype == motifcl::DType::F32 && spec.nbytes == C.nbytes();
            }
        }
        if (!saw_c_spec) return 20;

        auto plan = graph.plan_buffers();
        if (plan.allocations.empty() || plan.total_output_bytes == 0 || plan.peak_bytes == 0) return 21;
        if (plan.total_output_bytes < C.nbytes() + D.nbytes() + E.nbytes()) return 22;
        if (plan.peak_bytes > plan.total_output_bytes) return 23;

        auto optimized = graph.optimize();
        if (optimized.allocations.size() != plan.allocations.size()) return 24;
        auto polymorphic = graph.shape_polymorphic();
        motifcl::autograd::GraphOptimizeOptions poly_options;
        poly_options.shape_polymorphic = true;
        auto poly_plan = polymorphic.plan_buffers(poly_options);
        if (!poly_plan.shape_polymorphic) return 25;
        for (const auto& spec : polymorphic.tensor_specs_list()) {
            if (spec.dynamic_shape) saw_dynamic_spec = true;
        }
        if (!saw_dynamic_spec) return 26;
        if (!graph.compatible_with(graph.tensor_specs_list())) return 27;

        const auto& nodes = graph.nodes();
        if (nodes[0].op != "matmul_register_block4_f32" ||
            !same_ids(nodes[0].inputs, {A.id(), B.id()}) ||
            !same_ids(nodes[0].outputs, {C.id()})) return 5;
        if (nodes[1].op != "gelu_f32" ||
            !same_ids(nodes[1].inputs, {C.id()}) ||
            !same_ids(nodes[1].outputs, {D.id()})) return 6;
        if (nodes[2].op != "add_f32" ||
            !same_ids(nodes[2].inputs, {D.id(), D.id()}) ||
            !same_ids(nodes[2].outputs, {E.id()})) return 7;
        if (nodes[3].op != "view" ||
            !same_ids(nodes[3].inputs, {E.id()}) ||
            !same_ids(nodes[3].outputs, {V.id()})) return 8;

        auto values = V.to_vector<float>();
        if (values.size() != 4) return 9;
        const auto replay_ref = values;

        motifcl::scale_inplace(E, 0.0f);
        auto zeroed = V.to_vector<float>();
        for (float value : zeroed) {
            if (value != 0.0f) return 16;
        }

        graph.replay();
        auto replayed = V.to_vector<float>();
        for (std::size_t i = 0; i < replay_ref.size(); ++i) {
            if (std::fabs(replayed[i] - replay_ref[i]) > 1e-5f) return 17;
        }

        auto executor = ag::compile_graph_executor(graph);
        if (!executor.replayable() || !executor.runtime_plan().compatible || executor.runtime_plan().bindings.empty()) return 32;
        if (!executor.rebindable() || !executor.runtime_plan().kernel_rebinding) return 35;
        if (backend.command_buffer_mode().empty() || executor.execution_mode().empty()) return 40;
        if (backend.supports_command_buffer_mutable_dispatch() &&
            backend.command_buffer_mode() != "cl_khr_command_buffer_mutable_dispatch") return 41;
        motifcl::scale_inplace(E, 0.0f);
        executor.execute();
        if (executor.executions() != 1) return 33;
        auto executor_values = V.to_vector<float>();
        for (std::size_t i = 0; i < replay_ref.size(); ++i) {
            if (std::fabs(executor_values[i] - replay_ref[i]) > 1e-5f) return 34;
        }

        {
            std::vector<float> x_host = {1, 2, 3, 4};
            std::vector<float> y_host = {10, 20, 30, 40};
            auto X = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::F32, x_host.data());
            auto Y = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::F32, y_host.data());
            ag::GraphCaptureGuard rebind_guard;
            auto Z = motifcl::add(X, Y);
            auto rebind_graph = rebind_guard.finish();
            auto rebind_executor = ag::compile_graph_executor(rebind_graph);
            if (!rebind_executor.rebindable()) return 36;
            std::vector<float> x2_host = {100, 200, 300, 400};
            std::vector<float> y2_host = {1, 2, 3, 4};
            auto X2 = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::F32, x2_host.data());
            auto Y2 = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::F32, y2_host.data());
            rebind_executor.bind_tensor(X.id(), X2);
            rebind_executor.bind_tensor(Y.id(), Y2);
            rebind_executor.execute();
            auto rebound = Z.to_vector<float>();
            if (!same_ids({static_cast<int>(rebind_executor.bound_tensor_count())}, {2})) return 37;
            std::vector<float> rebound_ref = {101, 202, 303, 404};
            for (std::size_t i = 0; i < rebound.size(); ++i) {
                if (std::fabs(rebound[i] - rebound_ref[i]) > 1e-5f) return 38;
            }
            motifcl::scale_inplace(Z, 0.0f);
            rebind_executor.clear_bindings();
            rebind_executor.execute();
            auto original = Z.to_vector<float>();
            std::vector<float> original_ref = {11, 22, 33, 44};
            for (std::size_t i = 0; i < original.size(); ++i) {
                if (std::fabs(original[i] - original_ref[i]) > 1e-5f) return 39;
            }
        }

        {
            motifcl::nn::Linear linear(backend, 2, 1, false);
            motifcl::optim::Adam opt(linear.parameters(), 0.05f);
            std::vector<float> x_host = {1.0f, -1.0f, 0.5f, 2.0f};
            std::vector<float> target_host = {0.25f, -0.75f};
            auto X = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::F32, x_host.data());
            auto Target = motifcl::Tensor::from_cpu(backend, {2, 1}, motifcl::DType::F32, target_host.data());
            auto static_step = motifcl::train::capture_static_train_step([&]() {
                auto pred = linear.forward(X);
                auto loss = motifcl::mse_loss(pred, Target);
                loss.backward();
                opt.step();
                opt.zero_grad();
                return loss;
            });
            if (!static_step.replayable() || !static_step.runtime_plan().compatible) return 42;
            if (static_step.arena_binding_count() == 0 || static_step.arena_bytes() == 0) return 46;
            const float captured_loss = static_step.loss().item();
            static_step.execute();
            backend.finish();
            if (static_step.executions() != 1) return 43;
            const float replay_loss = static_step.loss().item();
            if (!std::isfinite(captured_loss) || !std::isfinite(replay_loss)) return 44;
            if (std::fabs(captured_loss - replay_loss) < 1e-7f) return 45;

            std::vector<float> x2_host = {-1.0f, 3.0f, 2.0f, -0.5f};
            std::vector<float> target2_host = {0.5f, 0.0f};
            auto X2 = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::F32, x2_host.data());
            auto Target2 = motifcl::Tensor::from_cpu(backend, {2, 1}, motifcl::DType::F32, target2_host.data());
            const float rebound_loss = static_step.execute_with_loss({
                {X.id(), X2},
                {Target.id(), Target2},
            });
            backend.finish();
            if (static_step.executions() != 2 || static_step.bound_tensor_count() != 2) return 47;
            if (!std::isfinite(rebound_loss)) return 48;
        }

        ag::GraphCaptureGuard guard;
        auto R = motifcl::relu(C);
        auto guarded = guard.finish();
        if (ag::is_graph_capturing()) return 10;
        if (guarded.size() != 1 ||
            guarded.nodes()[0].op != "relu_f32" ||
            !same_ids(guarded.nodes()[0].inputs, {C.id()}) ||
            !same_ids(guarded.nodes()[0].outputs, {R.id()})) return 11;

        bool threw = false;
        try {
            (void)guard.finish();
        } catch (const std::exception&) {
            threw = true;
        }
        if (!threw) return 12;

        {
            ag::GraphCaptureGuard discard;
            (void)motifcl::relu(C);
        }
        if (ag::is_graph_capturing()) return 13;

        ag::begin_graph_capture();
        auto loss = motifcl::mse_loss(C, D);
        auto loss_graph = ag::end_graph_capture();
        if (!loss_graph.replayable()) return 18;
        if (loss_graph.size() != 1 || loss_graph.nodes()[0].op != "mse_loss") return 28;
        motifcl::scale_inplace(C, 0.0f);
        loss_graph.replay();
        auto expected_loss = motifcl::mse_loss(C, D).item();
        if (std::fabs(loss.item() - expected_loss) > 1e-5f) return 29;

        std::vector<float> logits_values = {1.0f, 2.0f, -1.0f, 0.5f, 0.25f, 3.0f};
        std::vector<std::int32_t> target_values = {1, 2};
        auto Logits = motifcl::Tensor::from_cpu(backend, {2, 3}, motifcl::DType::F32, logits_values.data());
        auto Targets = motifcl::Tensor::from_cpu(backend, {2}, motifcl::DType::I32, target_values.data());
        ag::begin_graph_capture();
        auto ce_loss = motifcl::softmax_cross_entropy(Logits, Targets);
        auto ce_graph = ag::end_graph_capture();
        if (!ce_graph.replayable()) return 30;
        motifcl::scale_inplace(Logits, 0.0f);
        ce_graph.replay();
        auto expected_ce = motifcl::softmax_cross_entropy(Logits, Targets).item();
        if (std::fabs(ce_loss.item() - expected_ce) > 1e-5f) return 31;

        return 0;
    } catch (const std::exception& e) {
        ag::clear_graph_capture();
        return motifcl_test::handle_exception(e);
    }
}
