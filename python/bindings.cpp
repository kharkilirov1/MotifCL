#include <cstdint>
#include <memory>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <motifcl/motifcl.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_motifcl, m) {
    py::enum_<motifcl::DType>(m, "DType")
        .value("F32", motifcl::DType::F32)
        .value("I32", motifcl::DType::I32)
        .value("U8", motifcl::DType::U8)
        .value("F16", motifcl::DType::F16)
        .value("Q8_0", motifcl::DType::Q8_0)
        .value("Q4_0", motifcl::DType::Q4_0);

    py::class_<motifcl::DeviceInfo>(m, "DeviceInfo")
        .def_readonly("platform_name", &motifcl::DeviceInfo::platform_name)
        .def_readonly("platform_vendor", &motifcl::DeviceInfo::platform_vendor)
        .def_readonly("device_name", &motifcl::DeviceInfo::device_name)
        .def_readonly("device_vendor", &motifcl::DeviceInfo::device_vendor)
        .def_readonly("driver_version", &motifcl::DeviceInfo::driver_version)
        .def_readonly("device_version", &motifcl::DeviceInfo::device_version)
        .def_readonly("extensions", &motifcl::DeviceInfo::extensions)
        .def_readonly("global_mem_size", &motifcl::DeviceInfo::global_mem_size)
        .def_readonly("local_mem_size", &motifcl::DeviceInfo::local_mem_size)
        .def_readonly("compute_units", &motifcl::DeviceInfo::compute_units)
        .def_readonly("max_work_group_size", &motifcl::DeviceInfo::max_work_group_size);

    py::class_<motifcl::ProfileRecord>(m, "ProfileRecord")
        .def_readonly("name", &motifcl::ProfileRecord::name)
        .def_readonly("ms", &motifcl::ProfileRecord::ms);

    py::class_<motifcl::ProfileSummary>(m, "ProfileSummary")
        .def_readonly("name", &motifcl::ProfileSummary::name)
        .def_readonly("count", &motifcl::ProfileSummary::count)
        .def_readonly("total_ms", &motifcl::ProfileSummary::total_ms)
        .def_readonly("avg_ms", &motifcl::ProfileSummary::avg_ms)
        .def_readonly("max_ms", &motifcl::ProfileSummary::max_ms);

    py::class_<motifcl::Backend>(m, "Backend")
        .def_static("opencl", [](const std::string& kernel_dir) { return motifcl::Backend::create_opencl(kernel_dir); }, py::arg("kernel_dir") = "")
        .def_static("create", []() { return motifcl::Backend::create_opencl(); })
        .def("finish", &motifcl::Backend::finish)
        .def("device_info", &motifcl::Backend::device_info)
        .def("supports_integer_dot", &motifcl::Backend::supports_integer_dot)
        .def("int_dot_mode", &motifcl::Backend::int_dot_mode)
        .def("profile_set_enabled", [](motifcl::Backend& b, bool enabled) { b.profiler.set_enabled(enabled); })
        .def("profile_enabled", [](const motifcl::Backend& b) { return b.profiler.enabled(); })
        .def("profile_clear", [](motifcl::Backend& b) { b.profiler.clear(); })
        .def("profile_records", [](const motifcl::Backend& b) { return b.profiler.records(); })
        .def("profile_summary", [](const motifcl::Backend& b) { return b.profiler.summary(); });

    py::class_<motifcl::Tensor>(m, "Tensor")
        .def("id", &motifcl::Tensor::id)
        .def("cpu", [](const motifcl::Tensor& t) { return t.to_vector<float>(); })
        .def("cpu_i32", [](const motifcl::Tensor& t) { return t.to_vector<std::int32_t>(); })
        .def("item", &motifcl::Tensor::item)
        .def("quant_scale", &motifcl::Tensor::quant_scale)
        .def("has_quant_scales", &motifcl::Tensor::has_quant_scales)
        .def("quant_scales", &motifcl::Tensor::quant_scales)
        .def("quant_scale_axis", &motifcl::Tensor::quant_scale_axis)
        .def("quant_block_size", &motifcl::Tensor::quant_block_size)
        .def("backward", py::overload_cast<>(&motifcl::Tensor::backward))
        .def("set_requires_grad", &motifcl::Tensor::set_requires_grad, py::arg("value") = true)
        .def("view", [](const motifcl::Tensor& t, std::vector<int64_t> shape) { return t.view(motifcl::Shape(std::move(shape))); })
        .def_property_readonly("shape", [](const motifcl::Tensor& t) { return t.shape().dims; })
        .def_property_readonly("dtype", &motifcl::Tensor::dtype)
        .def_property_readonly("requires_grad", &motifcl::Tensor::requires_grad);

    py::class_<motifcl::autograd::GraphNodeInfo>(m, "GraphNodeInfo")
        .def_readonly("op", &motifcl::autograd::GraphNodeInfo::op)
        .def_readonly("inputs", &motifcl::autograd::GraphNodeInfo::inputs)
        .def_readonly("outputs", &motifcl::autograd::GraphNodeInfo::outputs)
        .def_readonly("temporaries", &motifcl::autograd::GraphNodeInfo::temporaries)
        .def_readonly("replayable", &motifcl::autograd::GraphNodeInfo::replayable);

    py::class_<motifcl::autograd::TensorSpec>(m, "TensorSpec")
        .def_readonly("id", &motifcl::autograd::TensorSpec::id)
        .def_property_readonly("shape", [](const motifcl::autograd::TensorSpec& spec) {
            return spec.shape.dims;
        })
        .def_readonly("dtype", &motifcl::autograd::TensorSpec::dtype)
        .def_readonly("nbytes", &motifcl::autograd::TensorSpec::nbytes)
        .def_readonly("dynamic_shape", &motifcl::autograd::TensorSpec::dynamic_shape);

    py::class_<motifcl::autograd::GraphBufferAllocation>(m, "GraphBufferAllocation")
        .def_readonly("allocation_id", &motifcl::autograd::GraphBufferAllocation::allocation_id)
        .def_readonly("tensor_ids", &motifcl::autograd::GraphBufferAllocation::tensor_ids)
        .def_readonly("bytes", &motifcl::autograd::GraphBufferAllocation::bytes)
        .def_readonly("first_node", &motifcl::autograd::GraphBufferAllocation::first_node)
        .def_readonly("last_node", &motifcl::autograd::GraphBufferAllocation::last_node);

    py::class_<motifcl::autograd::GraphBufferPlan>(m, "GraphBufferPlan")
        .def_readonly("allocations", &motifcl::autograd::GraphBufferPlan::allocations)
        .def_readonly("peak_bytes", &motifcl::autograd::GraphBufferPlan::peak_bytes)
        .def_readonly("total_output_bytes", &motifcl::autograd::GraphBufferPlan::total_output_bytes)
        .def_readonly("reused_bytes", &motifcl::autograd::GraphBufferPlan::reused_bytes)
        .def_readonly("shape_polymorphic", &motifcl::autograd::GraphBufferPlan::shape_polymorphic);

    py::class_<motifcl::autograd::GraphRuntimeBinding>(m, "GraphRuntimeBinding")
        .def_readonly("tensor_id", &motifcl::autograd::GraphRuntimeBinding::tensor_id)
        .def_readonly("allocation_id", &motifcl::autograd::GraphRuntimeBinding::allocation_id)
        .def_readonly("offset", &motifcl::autograd::GraphRuntimeBinding::offset)
        .def_readonly("nbytes", &motifcl::autograd::GraphRuntimeBinding::nbytes)
        .def_property_readonly("shape", [](const motifcl::autograd::GraphRuntimeBinding& b) { return b.shape.dims; })
        .def_readonly("dtype", &motifcl::autograd::GraphRuntimeBinding::dtype);

    py::class_<motifcl::autograd::GraphRuntimePlan>(m, "GraphRuntimePlan")
        .def_readonly("buffer_plan", &motifcl::autograd::GraphRuntimePlan::buffer_plan)
        .def_readonly("runtime_specs", &motifcl::autograd::GraphRuntimePlan::runtime_specs)
        .def_readonly("bindings", &motifcl::autograd::GraphRuntimePlan::bindings)
        .def_readonly("compatible", &motifcl::autograd::GraphRuntimePlan::compatible)
        .def_readonly("kernel_rebinding", &motifcl::autograd::GraphRuntimePlan::kernel_rebinding)
        .def_readonly("note", &motifcl::autograd::GraphRuntimePlan::note);

    py::class_<motifcl::autograd::GraphOptimizeOptions>(m, "GraphOptimizeOptions")
        .def(py::init<>())
        .def_readwrite("enable_buffer_reuse", &motifcl::autograd::GraphOptimizeOptions::enable_buffer_reuse)
        .def_readwrite("shape_polymorphic", &motifcl::autograd::GraphOptimizeOptions::shape_polymorphic);

    py::class_<motifcl::autograd::CapturedGraph>(m, "CapturedGraph")
        .def("size", &motifcl::autograd::CapturedGraph::size)
        .def("empty", &motifcl::autograd::CapturedGraph::empty)
        .def("replayable", &motifcl::autograd::CapturedGraph::replayable)
        .def("schedule", &motifcl::autograd::CapturedGraph::schedule)
        .def("tensor_specs_list", &motifcl::autograd::CapturedGraph::tensor_specs_list)
        .def("plan_buffers", [](const motifcl::autograd::CapturedGraph& g, const motifcl::autograd::GraphOptimizeOptions& options) {
            return g.plan_buffers(options);
        }, py::arg("options") = motifcl::autograd::GraphOptimizeOptions{})
        .def("optimize", [](const motifcl::autograd::CapturedGraph& g, const motifcl::autograd::GraphOptimizeOptions& options) {
            return g.optimize(options);
        }, py::arg("options") = motifcl::autograd::GraphOptimizeOptions{})
        .def("compile_runtime_plan", [](const motifcl::autograd::CapturedGraph& g,
                                        const std::vector<motifcl::autograd::TensorSpec>& specs,
                                        const motifcl::autograd::GraphOptimizeOptions& options) {
            return g.compile_runtime_plan(specs, options);
        }, py::arg("runtime_specs"), py::arg("options") = motifcl::autograd::GraphOptimizeOptions{})
        .def("shape_polymorphic", &motifcl::autograd::CapturedGraph::shape_polymorphic)
        .def("compatible_with", &motifcl::autograd::CapturedGraph::compatible_with, py::arg("runtime_specs"), py::arg("allow_dynamic_dims") = true)
        .def("replay", &motifcl::autograd::CapturedGraph::replay)
        .def("execute", &motifcl::autograd::CapturedGraph::execute)
        .def("clear", &motifcl::autograd::CapturedGraph::clear)
        .def_property_readonly("nodes", [](const motifcl::autograd::CapturedGraph& g) {
            return g.nodes();
        })
        .def_property_readonly("tensor_specs", [](const motifcl::autograd::CapturedGraph& g) {
            return g.tensor_specs_list();
        });

    py::class_<motifcl::autograd::GraphExecutor>(m, "GraphExecutor")
        .def("execute", &motifcl::autograd::GraphExecutor::execute)
        .def("replayable", &motifcl::autograd::GraphExecutor::replayable)
        .def("executions", &motifcl::autograd::GraphExecutor::executions)
        .def_property_readonly("runtime_plan", [](const motifcl::autograd::GraphExecutor& executor) -> const motifcl::autograd::GraphRuntimePlan& {
            return executor.runtime_plan();
        }, py::return_value_policy::reference_internal);

    py::class_<motifcl::autograd::GraphCaptureGuard>(m, "GraphCaptureGuard")
        .def(py::init<>())
        .def("finish", &motifcl::autograd::GraphCaptureGuard::finish);

    m.def("begin_graph_capture", &motifcl::autograd::begin_graph_capture);
    m.def("end_graph_capture", &motifcl::autograd::end_graph_capture);
    m.def("compile_graph_executor", [](const motifcl::autograd::CapturedGraph& graph,
                                       const motifcl::autograd::GraphOptimizeOptions& options) {
        return motifcl::autograd::compile_graph_executor(graph, options);
    }, py::arg("graph"), py::arg("options") = motifcl::autograd::GraphOptimizeOptions{});
    m.def("is_graph_capturing", &motifcl::autograd::is_graph_capturing);
    m.def("current_graph_capture", []() {
        return motifcl::autograd::current_graph_capture();
    });
    m.def("clear_graph_capture", &motifcl::autograd::clear_graph_capture);
    m.def("manual_seed", &motifcl::manual_seed);
    m.def("clear_memory_pool", &motifcl::clear_memory_pool);
    m.def("memory_pool_cached_blocks", &motifcl::memory_pool_cached_blocks);
    m.def("memory_pool_cached_bytes", &motifcl::memory_pool_cached_bytes);

    m.def("randn", [](motifcl::Backend& b, std::vector<int64_t> shape, float std) {
        return motifcl::Tensor::randn(b, motifcl::Shape(std::move(shape)), std);
    }, py::arg("backend"), py::arg("shape"), py::arg("std") = 0.02f);

    m.def("zeros", [](motifcl::Backend& b, std::vector<int64_t> shape, motifcl::DType dtype) {
        return motifcl::Tensor::zeros(b, motifcl::Shape(std::move(shape)), dtype);
    }, py::arg("backend"), py::arg("shape"), py::arg("dtype") = motifcl::DType::F32);

    m.def("ones", [](motifcl::Backend& b, std::vector<int64_t> shape, motifcl::DType dtype) {
        return motifcl::Tensor::ones(b, motifcl::Shape(std::move(shape)), dtype);
    }, py::arg("backend"), py::arg("shape"), py::arg("dtype") = motifcl::DType::F32);

    m.def("uniform", [](motifcl::Backend& b, std::vector<int64_t> shape, float low, float high) {
        return motifcl::Tensor::uniform(b, motifcl::Shape(std::move(shape)), low, high);
    }, py::arg("backend"), py::arg("shape"), py::arg("low") = -1.0f, py::arg("high") = 1.0f);

    m.def("tensor_f32", [](motifcl::Backend& b, std::vector<int64_t> shape, const std::vector<float>& values) {
        return motifcl::Tensor::from_cpu(b, motifcl::Shape(std::move(shape)), motifcl::DType::F32, values.data());
    });

    m.def("tensor_i32", [](motifcl::Backend& b, std::vector<int64_t> shape, const std::vector<std::int32_t>& values) {
        return motifcl::Tensor::from_cpu(b, motifcl::Shape(std::move(shape)), motifcl::DType::I32, values.data());
    });

    m.def("matmul", &motifcl::matmul);
    m.def("matmul_tiled_variant", &motifcl::matmul_tiled_variant, py::arg("a"), py::arg("b"), py::arg("tile"));
    m.def("backend_supports_fp16", &motifcl::backend_supports_fp16, py::arg("backend"));
    m.def("cast_f32_to_f16", &motifcl::cast_f32_to_f16);
    m.def("cast_f16_to_f32", &motifcl::cast_f16_to_f32);
    m.def("matmul_f16_accum_f32", &motifcl::matmul_f16_accum_f32);
    m.def("quantize_q8_symmetric", &motifcl::quantize_q8_symmetric, py::arg("x"), py::arg("scale") = 0.0f);
    m.def("dequantize_q8", &motifcl::dequantize_q8);
    m.def("quantize_q4_symmetric", &motifcl::quantize_q4_symmetric, py::arg("x"), py::arg("scale") = 0.0f);
    m.def("dequantize_q4", &motifcl::dequantize_q4);
    m.def("quantize_q8_symmetric_axis", &motifcl::quantize_q8_symmetric_axis, py::arg("x"), py::arg("axis"));
    m.def("quantize_q8_symmetric_rows", &motifcl::quantize_q8_symmetric_rows);
    m.def("quantize_q8_symmetric_cols", &motifcl::quantize_q8_symmetric_cols);
    m.def("quantize_q8_symmetric_blocks", &motifcl::quantize_q8_symmetric_blocks, py::arg("x"), py::arg("block_size"));
    m.def("quantize_q4_symmetric_axis", &motifcl::quantize_q4_symmetric_axis, py::arg("x"), py::arg("axis"));
    m.def("quantize_q4_symmetric_rows", &motifcl::quantize_q4_symmetric_rows);
    m.def("quantize_q4_symmetric_cols", &motifcl::quantize_q4_symmetric_cols);
    m.def("quantize_q4_symmetric_blocks", &motifcl::quantize_q4_symmetric_blocks, py::arg("x"), py::arg("block_size"));
    m.def("add", &motifcl::add);
    m.def("add_broadcast", &motifcl::add_broadcast);
    m.def("add_bias_gelu_rows", &motifcl::add_bias_gelu_rows);
    m.def("sub", &motifcl::sub);
    m.def("mul", &motifcl::mul);
    m.def("mul_broadcast", &motifcl::mul_broadcast);
    m.def("scale", &motifcl::scale);
    m.def("scale_inplace", &motifcl::scale_inplace);
    m.def("relu", &motifcl::relu);
    m.def("gelu", &motifcl::gelu);
    m.def("softmax_rows", &motifcl::softmax_rows);
    m.def("sum_rows", &motifcl::sum_rows);
    m.def("sum_all", &motifcl::sum_all);
    m.def("mean_all", &motifcl::mean_all);
    m.def("slice_rows", &motifcl::slice_rows);
    m.def("dropout", &motifcl::dropout, py::arg("x"), py::arg("p") = 0.5f, py::arg("training") = true);
    m.def("masked_fill", &motifcl::masked_fill, py::arg("x"), py::arg("mask"), py::arg("value"));
    m.def("rowwise_sum", &motifcl::rowwise_sum);
    m.def("rowwise_max", &motifcl::rowwise_max);
    m.def("multihead_attention", &motifcl::multihead_attention,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("n_head"),
          py::arg("causal") = false, py::arg("batch_size") = 1, py::arg("seq_len") = 0);
    m.def("rmsnorm", &motifcl::rmsnorm, py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-6f);
    m.def("layernorm", &motifcl::layernorm, py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-6f);
    m.def("mse_loss", &motifcl::mse_loss);
    m.def("softmax_cross_entropy", &motifcl::softmax_cross_entropy);
    m.def("save_tensor", &motifcl::save_tensor);
    m.def("load_tensor", &motifcl::load_tensor);
    m.def("save_parameters", &motifcl::save_parameters);
    m.def("load_parameters", &motifcl::load_parameters);

    py::class_<motifcl::nn::Parameter>(m, "Parameter")
        .def_property_readonly("data", [](motifcl::nn::Parameter& p) { return p.data; })
        .def("zero_grad", &motifcl::nn::Parameter::zero_grad);

    py::class_<motifcl::nn::Module, std::shared_ptr<motifcl::nn::Module>>(m, "Module")
        .def("forward", &motifcl::nn::Module::forward)
        .def("zero_grad", &motifcl::nn::Module::zero_grad);

    py::class_<motifcl::nn::Linear, motifcl::nn::Module, std::shared_ptr<motifcl::nn::Linear>>(m, "Linear")
        .def(py::init<motifcl::Backend&, int, int, bool>(), py::arg("backend"), py::arg("in_features"), py::arg("out_features"), py::arg("bias") = true)
        .def("forward", &motifcl::nn::Linear::forward)
        .def("parameters", &motifcl::nn::Linear::parameters, py::return_value_policy::reference);

    py::class_<motifcl::nn::GELU, motifcl::nn::Module, std::shared_ptr<motifcl::nn::GELU>>(m, "GELU")
        .def(py::init<>())
        .def("forward", &motifcl::nn::GELU::forward);

    py::class_<motifcl::nn::Sequential, motifcl::nn::Module, std::shared_ptr<motifcl::nn::Sequential>>(m, "Sequential")
        .def(py::init<std::vector<std::shared_ptr<motifcl::nn::Module>>>())
        .def("add", &motifcl::nn::Sequential::add)
        .def("forward", &motifcl::nn::Sequential::forward)
        .def("parameters", &motifcl::nn::Sequential::parameters, py::return_value_policy::reference);

    py::class_<motifcl::nn::GPTModel, motifcl::nn::Module, std::shared_ptr<motifcl::nn::GPTModel>>(m, "GPTModel")
        .def(py::init<motifcl::Backend&, int, int, int, int, int, int>(), py::arg("backend"), py::arg("vocab_size"), py::arg("block_size"), py::arg("n_embd"), py::arg("n_head"), py::arg("n_layer"), py::arg("mlp_hidden"))
        .def("forward", &motifcl::nn::GPTModel::forward)
        .def("parameters", &motifcl::nn::GPTModel::parameters, py::return_value_policy::reference);

    py::class_<motifcl::optim::Adam>(m, "Adam")
        .def(py::init<std::vector<motifcl::nn::Parameter*>, float, float, float, float>(), py::arg("params"), py::arg("lr") = 1e-3f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f)
        .def("step", &motifcl::optim::Adam::step)
        .def("zero_grad", &motifcl::optim::Adam::zero_grad)
        .def("set_lr", &motifcl::optim::Adam::set_lr)
        .def("lr", &motifcl::optim::Adam::lr);

    py::class_<motifcl::optim::SGD>(m, "SGD")
        .def(py::init<std::vector<motifcl::nn::Parameter*>, float>(), py::arg("params"), py::arg("lr") = 1e-2f)
        .def("step", &motifcl::optim::SGD::step)
        .def("zero_grad", &motifcl::optim::SGD::zero_grad)
        .def("set_lr", &motifcl::optim::SGD::set_lr)
        .def("lr", &motifcl::optim::SGD::lr);
}
