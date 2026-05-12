#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <motifcl/autograd/node.hpp>
#include <motifcl/motifcl.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_motifcl, m) {
    py::enum_<motifcl::DType>(m, "DType")
        .value("F32", motifcl::DType::F32)
        .value("I32", motifcl::DType::I32)
        .value("U8", motifcl::DType::U8)
        .value("F16", motifcl::DType::F16)
        .value("Q8_0", motifcl::DType::Q8_0)
        .value("Q4_0", motifcl::DType::Q4_0)
        .value("Q4_K", motifcl::DType::Q4_K)
        .value("Q5_K", motifcl::DType::Q5_K)
        .value("Q6_K", motifcl::DType::Q6_K)
        .value("Q4_0_COL", motifcl::DType::Q4_0_COL);

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
        .def("supports_command_buffer", &motifcl::Backend::supports_command_buffer)
        .def("supports_command_buffer_mutable_dispatch", &motifcl::Backend::supports_command_buffer_mutable_dispatch)
        .def("command_buffer_mode", &motifcl::Backend::command_buffer_mode)
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

    m.def("is_grad_enabled", &motifcl::autograd::is_enabled);
    m.def("set_grad_enabled", &motifcl::autograd::set_enabled, py::arg("enabled"));

    py::class_<motifcl::QKV>(m, "QKV")
        .def_readonly("q", &motifcl::QKV::q)
        .def_readonly("k", &motifcl::QKV::k)
        .def_readonly("v", &motifcl::QKV::v);

    py::class_<motifcl::autograd::GraphNodeInfo>(m, "GraphNodeInfo")
        .def_readonly("op", &motifcl::autograd::GraphNodeInfo::op)
        .def_readonly("inputs", &motifcl::autograd::GraphNodeInfo::inputs)
        .def_readonly("outputs", &motifcl::autograd::GraphNodeInfo::outputs)
        .def_readonly("temporaries", &motifcl::autograd::GraphNodeInfo::temporaries)
        .def_readonly("replayable", &motifcl::autograd::GraphNodeInfo::replayable)
        .def_readonly("rebindable", &motifcl::autograd::GraphNodeInfo::rebindable);

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
        .def("rebindable", &motifcl::autograd::CapturedGraph::rebindable)
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
        .def("replay", [](const motifcl::autograd::CapturedGraph& g) { g.replay(); })
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
        .def("rebindable", &motifcl::autograd::GraphExecutor::rebindable)
        .def("bind_tensor", &motifcl::autograd::GraphExecutor::bind_tensor,
             py::arg("captured_tensor_id"), py::arg("tensor"))
        .def("clear_bindings", &motifcl::autograd::GraphExecutor::clear_bindings)
        .def("bound_tensor_count", &motifcl::autograd::GraphExecutor::bound_tensor_count)
        .def("executions", &motifcl::autograd::GraphExecutor::executions)
        .def("execution_mode", &motifcl::autograd::GraphExecutor::execution_mode)
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
    m.def("swiglu", &motifcl::swiglu);
    m.def("softmax_rows", &motifcl::softmax_rows);
    m.def("sum_rows", &motifcl::sum_rows);
    m.def("sum_all", &motifcl::sum_all);
    m.def("mean_all", &motifcl::mean_all);
    m.def("slice_rows", &motifcl::slice_rows);
    m.def("dropout", &motifcl::dropout, py::arg("x"), py::arg("p") = 0.5f, py::arg("training") = true);
    m.def("masked_fill", &motifcl::masked_fill, py::arg("x"), py::arg("mask"), py::arg("value"));
    m.def("rowwise_sum", &motifcl::rowwise_sum);
    m.def("rowwise_max", &motifcl::rowwise_max);
    m.def("rowwise_argmax", &motifcl::rowwise_argmax);
    m.def("rowwise_sample", [](const motifcl::Tensor& x, float temperature, int top_k, float top_p, std::uint32_t seed) {
        return motifcl::rowwise_sample_top_p(x, temperature, top_k, top_p, seed);
    }, py::arg("x"), py::arg("temperature") = 0.0f, py::arg("top_k") = 0,
       py::arg("top_p") = 1.0f, py::arg("seed") = 1234);
    m.def("multihead_attention", &motifcl::multihead_attention,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("n_head"),
          py::arg("causal") = false, py::arg("batch_size") = 1, py::arg("seq_len") = 0);
    m.def("qkv_split", &motifcl::qkv_split,
          py::arg("packed"), py::arg("q_dim"), py::arg("kv_dim"));
    m.def("rope", &motifcl::rope,
          py::arg("x"), py::arg("n_head"), py::arg("batch_size"), py::arg("seq_len"),
          py::arg("theta") = 10000.0f, py::arg("rotary_dim") = 0, py::arg("token_offset") = 0);
    m.def("rope_positions", &motifcl::rope_positions,
          py::arg("x"), py::arg("positions"), py::arg("n_head"), py::arg("batch_size"), py::arg("seq_len"),
          py::arg("theta") = 10000.0f, py::arg("rotary_dim") = 0);
    m.def("grouped_query_attention", &motifcl::grouped_query_attention,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("n_head"), py::arg("n_kv_head"),
          py::arg("causal") = true, py::arg("batch_size") = 1, py::arg("query_len") = 0,
          py::arg("key_len") = 0, py::arg("query_offset") = 0);
    m.def("grouped_query_attention_windowed", &motifcl::grouped_query_attention_windowed,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("n_head"), py::arg("n_kv_head"),
          py::arg("sliding_window"), py::arg("causal") = true, py::arg("batch_size") = 1,
          py::arg("query_len") = 0, py::arg("key_len") = 0, py::arg("query_offset") = 0);
    m.def("grouped_query_attention_masked", &motifcl::grouped_query_attention_masked,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("mask"),
          py::arg("n_head"), py::arg("n_kv_head"), py::arg("causal") = true,
          py::arg("batch_size") = 1, py::arg("query_len") = 0,
          py::arg("key_len") = 0, py::arg("query_offset") = 0,
          py::arg("additive_mask") = false);
    m.def("kv_cache_append", &motifcl::kv_cache_append,
          py::arg("new_k"), py::arg("new_v"), py::arg("cache_k"), py::arg("cache_v"),
          py::arg("batch_size"), py::arg("new_tokens"), py::arg("max_tokens"), py::arg("start_pos"));
    m.def("kv_cache_append_positions", &motifcl::kv_cache_append_positions,
          py::arg("new_k"), py::arg("new_v"), py::arg("positions"), py::arg("cache_k"), py::arg("cache_v"),
          py::arg("batch_size"), py::arg("new_tokens"), py::arg("max_tokens"));
    m.def("rmsnorm", &motifcl::rmsnorm, py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-6f);
    m.def("layernorm", &motifcl::layernorm, py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-6f);
    m.def("mse_loss", &motifcl::mse_loss);
    m.def("softmax_cross_entropy", &motifcl::softmax_cross_entropy);
    m.def("save_tensor", &motifcl::save_tensor);
    m.def("load_tensor", &motifcl::load_tensor);
    m.def("save_parameters", &motifcl::save_parameters);
    m.def("load_parameters", &motifcl::load_parameters);
    m.def("save_quantized_transformer_checkpoint", [](const motifcl::nn::ModernGPTModel& model, const std::string& dir, motifcl::DType qdtype) {
        motifcl::save_quantized_transformer_checkpoint(model, dir, qdtype);
    }, py::arg("model"), py::arg("dir"), py::arg("qdtype") = motifcl::DType::Q4_0);
    m.def("save_quantized_transformer_checkpoint_policy", [](const motifcl::nn::ModernGPTModel& model, const std::string& dir, const motifcl::nn::QuantizationPolicy& policy) {
        motifcl::save_quantized_transformer_checkpoint(model, dir, policy);
    }, py::arg("model"), py::arg("dir"), py::arg("policy"));
    m.def("load_quantized_transformer_checkpoint", &motifcl::load_quantized_transformer_checkpoint,
          py::arg("model"), py::arg("backend"), py::arg("dir"));

    py::class_<motifcl::SafeTensorInfo>(m, "SafeTensorInfo")
        .def_readonly("name", &motifcl::SafeTensorInfo::name)
        .def_readonly("dtype", &motifcl::SafeTensorInfo::dtype)
        .def_readonly("shape", &motifcl::SafeTensorInfo::shape)
        .def_readonly("data_begin", &motifcl::SafeTensorInfo::data_begin)
        .def_readonly("data_end", &motifcl::SafeTensorInfo::data_end);

    py::class_<motifcl::SafeTensorsFile>(m, "SafeTensorsFile")
        .def_static("open", &motifcl::SafeTensorsFile::open, py::arg("path"))
        .def("path", &motifcl::SafeTensorsFile::path)
        .def("tensor_names", &motifcl::SafeTensorsFile::tensor_names)
        .def("contains", &motifcl::SafeTensorsFile::contains, py::arg("name"))
        .def("tensor_info", &motifcl::SafeTensorsFile::tensor_info, py::arg("name"), py::return_value_policy::reference_internal)
        .def("load_tensor", &motifcl::SafeTensorsFile::load_tensor,
             py::arg("backend"), py::arg("name"), py::arg("force_f32") = true)
        .def("load_f32_vector", &motifcl::SafeTensorsFile::load_f32_vector, py::arg("name"));

    m.def("load_safetensors", &motifcl::load_safetensors,
          py::arg("backend"), py::arg("paths"), py::arg("force_f32") = true);

    py::class_<motifcl::nn::Parameter>(m, "Parameter")
        .def(py::init<motifcl::Tensor, bool>(), py::arg("tensor"), py::arg("trainable") = true)
        .def_property("data",
                      [](motifcl::nn::Parameter& p) { return p.data; },
                      [](motifcl::nn::Parameter& p, motifcl::Tensor tensor) {
                          p.data = std::move(tensor);
                          p.data.set_requires_grad(p.trainable);
                      })
        .def_readwrite("trainable", &motifcl::nn::Parameter::trainable)
        .def("grad", [](const motifcl::nn::Parameter& p) {
            auto g = p.grad();
            return g ? *g : motifcl::Tensor{};
        })
        .def("zero_grad", &motifcl::nn::Parameter::zero_grad);

    py::class_<motifcl::nn::Module, std::shared_ptr<motifcl::nn::Module>>(m, "Module")
        .def("forward", &motifcl::nn::Module::forward)
        .def("zero_grad", &motifcl::nn::Module::zero_grad);

    py::class_<motifcl::nn::Linear, motifcl::nn::Module, std::shared_ptr<motifcl::nn::Linear>>(m, "Linear")
        .def(py::init<motifcl::Backend&, int, int, bool>(), py::arg("backend"), py::arg("in_features"), py::arg("out_features"), py::arg("bias") = true)
        .def("forward", &motifcl::nn::Linear::forward)
        .def("parameters", &motifcl::nn::Linear::parameters, py::return_value_policy::reference)
        .def_readwrite("weight", &motifcl::nn::Linear::weight)
        .def_readwrite("bias", &motifcl::nn::Linear::bias)
        .def("has_bias", &motifcl::nn::Linear::has_bias)
        .def("enable_quantized_inference", &motifcl::nn::Linear::enable_quantized_inference,
             py::arg("qdtype") = motifcl::DType::Q4_0)
        .def("set_quantized_weight", &motifcl::nn::Linear::set_quantized_weight, py::arg("weight"))
        .def("disable_quantized_inference", &motifcl::nn::Linear::disable_quantized_inference)
        .def("quantized_inference_enabled", &motifcl::nn::Linear::quantized_inference_enabled)
        .def("quantized_weight_dtype", &motifcl::nn::Linear::quantized_weight_dtype)
        .def("quantized_weight", &motifcl::nn::Linear::quantized_weight, py::return_value_policy::reference_internal);

    py::class_<motifcl::nn::QuantizedLinear, motifcl::nn::Module, std::shared_ptr<motifcl::nn::QuantizedLinear>>(m, "QuantizedLinear")
        .def(py::init<const motifcl::Tensor&, motifcl::DType>(),
             py::arg("weight"), py::arg("qdtype") = motifcl::DType::Q4_0)
        .def(py::init<const motifcl::Tensor&, const motifcl::Tensor&, motifcl::DType>(),
             py::arg("weight"), py::arg("bias"), py::arg("qdtype") = motifcl::DType::Q4_0)
        .def_static("from_linear", &motifcl::nn::QuantizedLinear::from_linear,
                    py::arg("linear"), py::arg("qdtype") = motifcl::DType::Q4_0)
        .def("forward", &motifcl::nn::QuantizedLinear::forward)
        .def("parameters", &motifcl::nn::QuantizedLinear::parameters, py::return_value_policy::reference)
        .def("quantized_weight", &motifcl::nn::QuantizedLinear::quantized_weight, py::return_value_policy::reference_internal)
        .def("bias", &motifcl::nn::QuantizedLinear::bias, py::return_value_policy::reference_internal)
        .def("has_bias", &motifcl::nn::QuantizedLinear::has_bias)
        .def("weight_dtype", &motifcl::nn::QuantizedLinear::weight_dtype)
        .def("in_features", &motifcl::nn::QuantizedLinear::in_features)
        .def("out_features", &motifcl::nn::QuantizedLinear::out_features);

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

    py::class_<motifcl::nn::TransformerConfig>(m, "TransformerConfig")
        .def(py::init<>())
        .def_readwrite("vocab_size", &motifcl::nn::TransformerConfig::vocab_size)
        .def_readwrite("block_size", &motifcl::nn::TransformerConfig::block_size)
        .def_readwrite("n_embd", &motifcl::nn::TransformerConfig::n_embd)
        .def_readwrite("n_head", &motifcl::nn::TransformerConfig::n_head)
        .def_readwrite("n_kv_head", &motifcl::nn::TransformerConfig::n_kv_head)
        .def_readwrite("head_dim", &motifcl::nn::TransformerConfig::head_dim)
        .def_readwrite("n_layer", &motifcl::nn::TransformerConfig::n_layer)
        .def_readwrite("mlp_hidden", &motifcl::nn::TransformerConfig::mlp_hidden)
        .def_readwrite("dropout", &motifcl::nn::TransformerConfig::dropout)
        .def_readwrite("use_rope", &motifcl::nn::TransformerConfig::use_rope)
        .def_readwrite("use_swiglu", &motifcl::nn::TransformerConfig::use_swiglu)
        .def_readwrite("use_qkv_bias", &motifcl::nn::TransformerConfig::use_qkv_bias)
        .def_readwrite("causal", &motifcl::nn::TransformerConfig::causal)
        .def_readwrite("learned_position_embeddings", &motifcl::nn::TransformerConfig::learned_position_embeddings)
        .def_readwrite("rope_theta", &motifcl::nn::TransformerConfig::rope_theta)
        .def_readwrite("rotary_dim", &motifcl::nn::TransformerConfig::rotary_dim)
        .def_readwrite("sliding_window", &motifcl::nn::TransformerConfig::sliding_window)
        .def_readwrite("layer_head_dims", &motifcl::nn::TransformerConfig::layer_head_dims)
        .def_readwrite("split_qkv_projections", &motifcl::nn::TransformerConfig::split_qkv_projections)
        .def_readwrite("split_mlp_projections", &motifcl::nn::TransformerConfig::split_mlp_projections);

    py::class_<motifcl::nn::KVCache>(m, "KVCache")
        .def(py::init<>())
        .def(py::init<motifcl::Backend&, int64_t, int64_t, int, int>(),
             py::arg("backend"), py::arg("batch_size"), py::arg("max_seq_len"),
             py::arg("n_kv_head"), py::arg("head_dim"))
        .def_readwrite("k", &motifcl::nn::KVCache::k)
        .def_readwrite("v", &motifcl::nn::KVCache::v)
        .def_readwrite("batch_size", &motifcl::nn::KVCache::batch_size)
        .def_readwrite("max_seq_len", &motifcl::nn::KVCache::max_seq_len)
        .def_readwrite("length", &motifcl::nn::KVCache::length)
        .def_readwrite("n_kv_head", &motifcl::nn::KVCache::n_kv_head)
        .def_readwrite("head_dim", &motifcl::nn::KVCache::head_dim)
        .def("reset", &motifcl::nn::KVCache::reset);

    py::class_<motifcl::nn::PagedKVCache>(m, "PagedKVCache")
        .def(py::init<>())
        .def(py::init<motifcl::Backend&, int64_t, int64_t, int64_t, int, int>(),
             py::arg("backend"), py::arg("batch_size"), py::arg("max_seq_len"),
             py::arg("page_size"), py::arg("n_kv_head"), py::arg("head_dim"))
        .def_readwrite("k_pages", &motifcl::nn::PagedKVCache::k_pages)
        .def_readwrite("v_pages", &motifcl::nn::PagedKVCache::v_pages)
        .def_readwrite("page_table", &motifcl::nn::PagedKVCache::page_table)
        .def_readwrite("batch_size", &motifcl::nn::PagedKVCache::batch_size)
        .def_readwrite("max_seq_len", &motifcl::nn::PagedKVCache::max_seq_len)
        .def_readwrite("page_size", &motifcl::nn::PagedKVCache::page_size)
        .def_readwrite("page_count", &motifcl::nn::PagedKVCache::page_count)
        .def_readwrite("length", &motifcl::nn::PagedKVCache::length)
        .def_readwrite("tokens_seen", &motifcl::nn::PagedKVCache::tokens_seen)
        .def_readwrite("n_kv_head", &motifcl::nn::PagedKVCache::n_kv_head)
        .def_readwrite("head_dim", &motifcl::nn::PagedKVCache::head_dim)
        .def("capacity", &motifcl::nn::PagedKVCache::capacity)
        .def("reset", &motifcl::nn::PagedKVCache::reset);

    py::class_<motifcl::nn::DeltaStateCache>(m, "DeltaStateCache")
        .def(py::init<>())
        .def(py::init<motifcl::Backend&, int64_t, int, int, int>(),
             py::arg("backend"), py::arg("batch_size"), py::arg("num_heads"),
             py::arg("head_dim"), py::arg("state_dim"))
        .def_readwrite("state", &motifcl::nn::DeltaStateCache::state)
        .def_readwrite("batch_size", &motifcl::nn::DeltaStateCache::batch_size)
        .def_readwrite("num_heads", &motifcl::nn::DeltaStateCache::num_heads)
        .def_readwrite("head_dim", &motifcl::nn::DeltaStateCache::head_dim)
        .def_readwrite("state_dim", &motifcl::nn::DeltaStateCache::state_dim)
        .def("zero", &motifcl::nn::DeltaStateCache::zero);

    py::class_<motifcl::nn::ModernMLP, motifcl::nn::Module, std::shared_ptr<motifcl::nn::ModernMLP>>(m, "ModernMLP")
        .def(py::init<motifcl::Backend&, int, int, bool, bool, float>(),
             py::arg("backend"), py::arg("n_embd"), py::arg("hidden"),
             py::arg("use_swiglu") = true, py::arg("use_bias") = false, py::arg("dropout") = 0.0f)
        .def("forward", &motifcl::nn::ModernMLP::forward)
        .def("parameters", &motifcl::nn::ModernMLP::parameters, py::return_value_policy::reference)
        .def("enable_quantized_inference", &motifcl::nn::ModernMLP::enable_quantized_inference,
             py::arg("qdtype") = motifcl::DType::Q4_0)
        .def("disable_quantized_inference", &motifcl::nn::ModernMLP::disable_quantized_inference)
        .def("quantized_inference_enabled", &motifcl::nn::ModernMLP::quantized_inference_enabled)
        .def("quantized_weight_dtype", &motifcl::nn::ModernMLP::quantized_weight_dtype)
        .def("enable_split_projections", &motifcl::nn::ModernMLP::enable_split_projections, py::arg("enabled") = true)
        .def("split_projections_enabled", &motifcl::nn::ModernMLP::split_projections_enabled)
        .def_readwrite("use_swiglu", &motifcl::nn::ModernMLP::use_swiglu)
        .def_readwrite("dropout_p", &motifcl::nn::ModernMLP::dropout_p);

    py::class_<motifcl::nn::MoEFFN, motifcl::nn::Module, std::shared_ptr<motifcl::nn::MoEFFN>>(m, "MoEFFN")
        .def(py::init<motifcl::Backend&, int, int, int, int>(),
             py::arg("backend"), py::arg("n_embd"), py::arg("hidden"),
             py::arg("num_experts"), py::arg("experts_per_token"))
        .def("forward", &motifcl::nn::MoEFFN::forward)
        .def("parameters", &motifcl::nn::MoEFFN::parameters, py::return_value_policy::reference)
        .def_readwrite("router_weight", &motifcl::nn::MoEFFN::router_weight)
        .def_readwrite("expert_gate_weight", &motifcl::nn::MoEFFN::expert_gate_weight)
        .def_readwrite("expert_up_weight", &motifcl::nn::MoEFFN::expert_up_weight)
        .def_readwrite("expert_down_weight", &motifcl::nn::MoEFFN::expert_down_weight)
        .def_readwrite("num_experts", &motifcl::nn::MoEFFN::num_experts)
        .def_readwrite("experts_per_token", &motifcl::nn::MoEFFN::experts_per_token);

    py::class_<motifcl::nn::GatedDeltaNetLayer, motifcl::nn::Module, std::shared_ptr<motifcl::nn::GatedDeltaNetLayer>>(m, "GatedDeltaNetLayer")
        .def(py::init<motifcl::Backend&, const motifcl::nn::TransformerConfig&, int>(),
             py::arg("backend"), py::arg("config"), py::arg("state_dim") = 0)
        .def("forward", &motifcl::nn::GatedDeltaNetLayer::forward)
        .def("forward_with_state", &motifcl::nn::GatedDeltaNetLayer::forward_with_state,
             py::arg("x"), py::arg("state"), py::arg("batch_size"), py::arg("seq_len"))
        .def("parameters", &motifcl::nn::GatedDeltaNetLayer::parameters, py::return_value_policy::reference)
        .def("n_head", &motifcl::nn::GatedDeltaNetLayer::n_head)
        .def("head_dim", &motifcl::nn::GatedDeltaNetLayer::head_dim)
        .def("state_dim", &motifcl::nn::GatedDeltaNetLayer::state_dim)
        .def("decay", &motifcl::nn::GatedDeltaNetLayer::decay)
        .def("set_decay", &motifcl::nn::GatedDeltaNetLayer::set_decay);

    py::class_<motifcl::nn::ModernSelfAttention, motifcl::nn::Module, std::shared_ptr<motifcl::nn::ModernSelfAttention>>(m, "ModernSelfAttention")
        .def(py::init<motifcl::Backend&, const motifcl::nn::TransformerConfig&>(),
             py::arg("backend"), py::arg("config"))
        .def("forward", py::overload_cast<const motifcl::Tensor&>(&motifcl::nn::ModernSelfAttention::forward))
        .def("forward", py::overload_cast<const motifcl::Tensor&, int64_t, int64_t, bool>(&motifcl::nn::ModernSelfAttention::forward),
             py::arg("x"), py::arg("batch_size"), py::arg("seq_len"), py::arg("causal") = true)
        .def("forward_masked", &motifcl::nn::ModernSelfAttention::forward_masked,
             py::arg("x"), py::arg("mask"), py::arg("batch_size"), py::arg("seq_len"), py::arg("causal") = true)
        .def("forward_with_cache", py::overload_cast<const motifcl::Tensor&, motifcl::nn::KVCache&, int64_t, int64_t>(&motifcl::nn::ModernSelfAttention::forward_with_cache),
             py::arg("x"), py::arg("cache"), py::arg("batch_size"), py::arg("seq_len"))
        .def("forward_with_paged_cache", py::overload_cast<const motifcl::Tensor&, motifcl::nn::PagedKVCache&, int64_t, int64_t>(&motifcl::nn::ModernSelfAttention::forward_with_cache),
             py::arg("x"), py::arg("cache"), py::arg("batch_size"), py::arg("seq_len"))
        .def("forward_with_cache_masked", &motifcl::nn::ModernSelfAttention::forward_with_cache_masked,
             py::arg("x"), py::arg("mask"), py::arg("cache"), py::arg("batch_size"), py::arg("seq_len"))
        .def("parameters", &motifcl::nn::ModernSelfAttention::parameters, py::return_value_policy::reference)
        .def("n_head", &motifcl::nn::ModernSelfAttention::n_head)
        .def("n_kv_head", &motifcl::nn::ModernSelfAttention::n_kv_head)
        .def("head_dim", &motifcl::nn::ModernSelfAttention::head_dim)
        .def("enable_quantized_inference", &motifcl::nn::ModernSelfAttention::enable_quantized_inference,
             py::arg("qdtype") = motifcl::DType::Q4_0)
        .def("disable_quantized_inference", &motifcl::nn::ModernSelfAttention::disable_quantized_inference)
        .def("quantized_inference_enabled", &motifcl::nn::ModernSelfAttention::quantized_inference_enabled)
        .def("quantized_weight_dtype", &motifcl::nn::ModernSelfAttention::quantized_weight_dtype)
        .def("enable_split_projections", &motifcl::nn::ModernSelfAttention::enable_split_projections, py::arg("enabled") = true)
        .def("split_projections_enabled", &motifcl::nn::ModernSelfAttention::split_projections_enabled);

    py::class_<motifcl::nn::GatedAttentionLayer, motifcl::nn::Module, std::shared_ptr<motifcl::nn::GatedAttentionLayer>>(m, "GatedAttentionLayer")
        .def(py::init<motifcl::Backend&, const motifcl::nn::TransformerConfig&>(),
             py::arg("backend"), py::arg("config"))
        .def("forward", py::overload_cast<const motifcl::Tensor&>(&motifcl::nn::GatedAttentionLayer::forward))
        .def("forward", py::overload_cast<const motifcl::Tensor&, int64_t, int64_t, bool>(&motifcl::nn::GatedAttentionLayer::forward),
             py::arg("x"), py::arg("batch_size"), py::arg("seq_len"), py::arg("causal") = true)
        .def("forward_with_cache", py::overload_cast<const motifcl::Tensor&, motifcl::nn::KVCache&, int64_t, int64_t>(&motifcl::nn::GatedAttentionLayer::forward_with_cache),
             py::arg("x"), py::arg("cache"), py::arg("batch_size"), py::arg("seq_len"))
        .def("forward_with_paged_cache", py::overload_cast<const motifcl::Tensor&, motifcl::nn::PagedKVCache&, int64_t, int64_t>(&motifcl::nn::GatedAttentionLayer::forward_with_cache),
             py::arg("x"), py::arg("cache"), py::arg("batch_size"), py::arg("seq_len"))
        .def("parameters", &motifcl::nn::GatedAttentionLayer::parameters, py::return_value_policy::reference);

    py::class_<motifcl::nn::ModernTransformerBlock, motifcl::nn::Module, std::shared_ptr<motifcl::nn::ModernTransformerBlock>>(m, "ModernTransformerBlock")
        .def(py::init<motifcl::Backend&, const motifcl::nn::TransformerConfig&>(),
             py::arg("backend"), py::arg("config"))
        .def("forward", py::overload_cast<const motifcl::Tensor&>(&motifcl::nn::ModernTransformerBlock::forward))
        .def("forward", py::overload_cast<const motifcl::Tensor&, int64_t, int64_t>(&motifcl::nn::ModernTransformerBlock::forward),
             py::arg("x"), py::arg("batch_size"), py::arg("seq_len"))
        .def("forward_masked", &motifcl::nn::ModernTransformerBlock::forward_masked,
             py::arg("x"), py::arg("mask"), py::arg("batch_size"), py::arg("seq_len"))
        .def("forward_with_cache", py::overload_cast<const motifcl::Tensor&, motifcl::nn::KVCache&, int64_t, int64_t>(&motifcl::nn::ModernTransformerBlock::forward_with_cache),
             py::arg("x"), py::arg("cache"), py::arg("batch_size"), py::arg("seq_len"))
        .def("forward_with_paged_cache", py::overload_cast<const motifcl::Tensor&, motifcl::nn::PagedKVCache&, int64_t, int64_t>(&motifcl::nn::ModernTransformerBlock::forward_with_cache),
             py::arg("x"), py::arg("cache"), py::arg("batch_size"), py::arg("seq_len"))
        .def("forward_with_cache_masked", &motifcl::nn::ModernTransformerBlock::forward_with_cache_masked,
             py::arg("x"), py::arg("mask"), py::arg("cache"), py::arg("batch_size"), py::arg("seq_len"))
        .def("parameters", &motifcl::nn::ModernTransformerBlock::parameters, py::return_value_policy::reference)
        .def("enable_quantized_inference", &motifcl::nn::ModernTransformerBlock::enable_quantized_inference,
             py::arg("qdtype") = motifcl::DType::Q4_0)
        .def("disable_quantized_inference", &motifcl::nn::ModernTransformerBlock::disable_quantized_inference)
        .def("quantized_inference_enabled", &motifcl::nn::ModernTransformerBlock::quantized_inference_enabled)
        .def("quantized_weight_dtype", &motifcl::nn::ModernTransformerBlock::quantized_weight_dtype);

    py::class_<motifcl::nn::QuantizationPolicy>(m, "QuantizationPolicy")
        .def(py::init<>())
        .def_readwrite("default_dtype", &motifcl::nn::QuantizationPolicy::default_dtype)
        .def_readwrite("lm_head_dtype", &motifcl::nn::QuantizationPolicy::lm_head_dtype)
        .def_readwrite("q8_layers", &motifcl::nn::QuantizationPolicy::q8_layers);

    py::class_<motifcl::nn::ModernGPTModel, motifcl::nn::Module, std::shared_ptr<motifcl::nn::ModernGPTModel>>(m, "ModernGPTModel")
        .def(py::init<motifcl::Backend&, const motifcl::nn::TransformerConfig&>(),
             py::arg("backend"), py::arg("config"))
        .def("forward", &motifcl::nn::ModernGPTModel::forward)
        .def("forward_masked", &motifcl::nn::ModernGPTModel::forward_masked,
             py::arg("token_ids"), py::arg("mask"))
        .def("forward_with_cache", [](motifcl::nn::ModernGPTModel& model,
                                      const motifcl::Tensor& token_ids,
                                      py::list cache_list) {
            std::vector<motifcl::nn::KVCache*> kv_caches;
            kv_caches.reserve(static_cast<std::size_t>(py::len(cache_list)));
            bool all_kv = true;
            for (py::handle item : cache_list) {
                try {
                    kv_caches.push_back(&item.cast<motifcl::nn::KVCache&>());
                } catch (const py::cast_error&) {
                    all_kv = false;
                    break;
                }
            }
            if (all_kv) return model.forward_with_cache(token_ids, kv_caches);
            std::vector<motifcl::nn::PagedKVCache*> paged_caches;
            paged_caches.reserve(static_cast<std::size_t>(py::len(cache_list)));
            for (py::handle item : cache_list) {
                paged_caches.push_back(&item.cast<motifcl::nn::PagedKVCache&>());
            }
            return model.forward_with_cache(token_ids, paged_caches);
        }, py::arg("token_ids"), py::arg("caches"))
        .def("forward_with_cache_masked", [](motifcl::nn::ModernGPTModel& model,
                                             const motifcl::Tensor& token_ids,
                                             const motifcl::Tensor& mask,
                                             py::list cache_list) {
            std::vector<motifcl::nn::KVCache*> caches;
            caches.reserve(static_cast<std::size_t>(py::len(cache_list)));
            for (py::handle item : cache_list) {
                caches.push_back(&item.cast<motifcl::nn::KVCache&>());
            }
            return model.forward_with_cache_masked(token_ids, mask, caches);
        }, py::arg("token_ids"), py::arg("mask"), py::arg("caches"))
        .def("parameters", &motifcl::nn::ModernGPTModel::parameters, py::return_value_policy::reference)
        .def("create_kv_cache", &motifcl::nn::ModernGPTModel::create_kv_cache,
             py::arg("backend"), py::arg("batch_size"))
        .def("create_paged_kv_cache", &motifcl::nn::ModernGPTModel::create_paged_kv_cache,
             py::arg("backend"), py::arg("batch_size"), py::arg("page_size") = 256)
        .def("create_delta_state_cache", &motifcl::nn::ModernGPTModel::create_delta_state_cache,
             py::arg("backend"), py::arg("batch_size"), py::arg("state_dim") = 0)
        .def("set_layer_attention_window", &motifcl::nn::ModernGPTModel::set_layer_attention_window,
             py::arg("layer"), py::arg("window"))
        .def("enable_quantized_inference", [](motifcl::nn::ModernGPTModel& model, motifcl::DType qdtype) {
            model.enable_quantized_inference(qdtype);
        }, py::arg("qdtype") = motifcl::DType::Q4_0)
        .def("enable_quantized_inference_policy", [](motifcl::nn::ModernGPTModel& model, const motifcl::nn::QuantizationPolicy& policy) {
            model.enable_quantized_inference(policy);
        }, py::arg("policy"))
        .def("set_quantized_lm_head", &motifcl::nn::ModernGPTModel::set_quantized_lm_head, py::arg("weight"))
        .def("disable_quantized_inference", &motifcl::nn::ModernGPTModel::disable_quantized_inference)
        .def("quantized_inference_enabled", &motifcl::nn::ModernGPTModel::quantized_inference_enabled)
        .def("quantized_weight_dtype", &motifcl::nn::ModernGPTModel::quantized_weight_dtype)
        .def("quantized_lm_head", &motifcl::nn::ModernGPTModel::quantized_lm_head, py::return_value_policy::reference_internal)
        .def_readwrite("config", &motifcl::nn::ModernGPTModel::config);

    py::enum_<motifcl::nn::HybridAttentionKind>(m, "HybridAttentionKind")
        .value("FullAttention", motifcl::nn::HybridAttentionKind::FullAttention)
        .value("SlidingAttention", motifcl::nn::HybridAttentionKind::SlidingAttention)
        .value("GatedAttention", motifcl::nn::HybridAttentionKind::GatedAttention)
        .value("GatedDeltaNet", motifcl::nn::HybridAttentionKind::GatedDeltaNet);

    py::enum_<motifcl::nn::HybridFFNKind>(m, "HybridFFNKind")
        .value("SwiGLUFFN", motifcl::nn::HybridFFNKind::SwiGLUFFN)
        .value("MoEFFN", motifcl::nn::HybridFFNKind::MoEFFN);

    py::class_<motifcl::nn::HybridLayerConfig>(m, "HybridLayerConfig")
        .def(py::init<>())
        .def_readwrite("attention", &motifcl::nn::HybridLayerConfig::attention)
        .def_readwrite("ffn", &motifcl::nn::HybridLayerConfig::ffn)
        .def_readwrite("sliding_window", &motifcl::nn::HybridLayerConfig::sliding_window)
        .def_readwrite("num_experts", &motifcl::nn::HybridLayerConfig::num_experts)
        .def_readwrite("experts_per_token", &motifcl::nn::HybridLayerConfig::experts_per_token)
        .def_readwrite("delta_state_dim", &motifcl::nn::HybridLayerConfig::delta_state_dim);

    py::class_<motifcl::nn::HybridRuntimeCache>(m, "HybridRuntimeCache")
        .def("reset", &motifcl::nn::HybridRuntimeCache::reset)
        .def_readwrite("use_paged_kv", &motifcl::nn::HybridRuntimeCache::use_paged_kv)
        .def_readwrite("batch_size", &motifcl::nn::HybridRuntimeCache::batch_size)
        .def_readwrite("page_size", &motifcl::nn::HybridRuntimeCache::page_size);

    py::class_<motifcl::nn::HybridGPTModel, motifcl::nn::Module, std::shared_ptr<motifcl::nn::HybridGPTModel>>(m, "HybridGPTModel")
        .def(py::init<motifcl::Backend&, const motifcl::nn::TransformerConfig&, const std::vector<motifcl::nn::HybridLayerConfig>&>(),
             py::arg("backend"), py::arg("config"), py::arg("layer_configs") = std::vector<motifcl::nn::HybridLayerConfig>{})
        .def("forward", &motifcl::nn::HybridGPTModel::forward)
        .def("forward_with_cache", &motifcl::nn::HybridGPTModel::forward_with_cache,
             py::arg("token_ids"), py::arg("cache"))
        .def("parameters", &motifcl::nn::HybridGPTModel::parameters, py::return_value_policy::reference)
        .def("create_runtime_cache", &motifcl::nn::HybridGPTModel::create_runtime_cache,
             py::arg("backend"), py::arg("batch_size"), py::arg("use_paged_kv") = false, py::arg("page_size") = 256)
        .def("set_quantized_lm_head", &motifcl::nn::HybridGPTModel::set_quantized_lm_head, py::arg("weight"))
        .def("disable_quantized_inference", &motifcl::nn::HybridGPTModel::disable_quantized_inference)
        .def("quantized_inference_enabled", &motifcl::nn::HybridGPTModel::quantized_inference_enabled)
        .def("quantized_weight_dtype", &motifcl::nn::HybridGPTModel::quantized_weight_dtype)
        .def("quantized_lm_head", &motifcl::nn::HybridGPTModel::quantized_lm_head, py::return_value_policy::reference_internal)
        .def_readwrite("config", &motifcl::nn::HybridGPTModel::config)
        .def_readwrite("layer_configs", &motifcl::nn::HybridGPTModel::layer_configs);

    py::class_<motifcl::nn::GemmaConfig>(m, "GemmaConfig")
        .def(py::init<>())
        .def_readwrite("vocab_size", &motifcl::nn::GemmaConfig::vocab_size)
        .def_readwrite("max_position_embeddings", &motifcl::nn::GemmaConfig::max_position_embeddings)
        .def_readwrite("hidden_size", &motifcl::nn::GemmaConfig::hidden_size)
        .def_readwrite("intermediate_size", &motifcl::nn::GemmaConfig::intermediate_size)
        .def_readwrite("num_hidden_layers", &motifcl::nn::GemmaConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &motifcl::nn::GemmaConfig::num_attention_heads)
        .def_readwrite("num_key_value_heads", &motifcl::nn::GemmaConfig::num_key_value_heads)
        .def_readwrite("head_dim", &motifcl::nn::GemmaConfig::head_dim)
        .def_readwrite("rms_norm_eps", &motifcl::nn::GemmaConfig::rms_norm_eps)
        .def_readwrite("rope_theta", &motifcl::nn::GemmaConfig::rope_theta)
        .def_readwrite("attention_dropout", &motifcl::nn::GemmaConfig::attention_dropout)
        .def_readwrite("attention_bias", &motifcl::nn::GemmaConfig::attention_bias)
        .def_readwrite("attention_k_eq_v", &motifcl::nn::GemmaConfig::attention_k_eq_v)
        .def_readwrite("tie_word_embeddings", &motifcl::nn::GemmaConfig::tie_word_embeddings)
        .def_readwrite("bos_token_id", &motifcl::nn::GemmaConfig::bos_token_id)
        .def_readwrite("eos_token_id", &motifcl::nn::GemmaConfig::eos_token_id)
        .def_readwrite("pad_token_id", &motifcl::nn::GemmaConfig::pad_token_id)
        .def_readwrite("sliding_window", &motifcl::nn::GemmaConfig::sliding_window);

    py::class_<motifcl::nn::GemmaWeightName>(m, "GemmaWeightName")
        .def_readonly("hf_name", &motifcl::nn::GemmaWeightName::hf_name)
        .def_readonly("internal_name", &motifcl::nn::GemmaWeightName::internal_name)
        .def_readonly("kind", &motifcl::nn::GemmaWeightName::kind)
        .def_readonly("layer", &motifcl::nn::GemmaWeightName::layer);

    py::class_<motifcl::nn::GemmaWeightLoadReport>(m, "GemmaWeightLoadReport")
        .def_readonly("loaded_tensors", &motifcl::nn::GemmaWeightLoadReport::loaded_tensors)
        .def_readonly("applied", &motifcl::nn::GemmaWeightLoadReport::applied)
        .def_readonly("missing", &motifcl::nn::GemmaWeightLoadReport::missing)
        .def_readonly("unexpected", &motifcl::nn::GemmaWeightLoadReport::unexpected);

    py::class_<motifcl::nn::GemmaTokenizer>(m, "GemmaTokenizer")
        .def(py::init<>())
        .def_static("byte_fallback", &motifcl::nn::GemmaTokenizer::byte_fallback,
                    py::arg("vocab_size") = 256, py::arg("bos_token_id") = 1, py::arg("eos_token_id") = 2)
        .def_static("load_vocab", &motifcl::nn::GemmaTokenizer::load_vocab,
                    py::arg("path"), py::arg("bos_token_id") = 1, py::arg("eos_token_id") = 2)
        .def("encode", &motifcl::nn::GemmaTokenizer::encode,
             py::arg("text"), py::arg("add_bos") = false, py::arg("add_eos") = false)
        .def("decode", &motifcl::nn::GemmaTokenizer::decode,
             py::arg("ids"), py::arg("skip_special") = true)
        .def("vocab_size", &motifcl::nn::GemmaTokenizer::vocab_size)
        .def("bos_token_id", &motifcl::nn::GemmaTokenizer::bos_token_id)
        .def("eos_token_id", &motifcl::nn::GemmaTokenizer::eos_token_id)
        .def("is_byte_fallback", &motifcl::nn::GemmaTokenizer::is_byte_fallback)
        .def("tokenizer_model_type", &motifcl::nn::GemmaTokenizer::tokenizer_model_type);

    py::class_<motifcl::nn::GenerateOptions>(m, "GenerateOptions")
        .def(py::init<>())
        .def_readwrite("max_new_tokens", &motifcl::nn::GenerateOptions::max_new_tokens)
        .def_readwrite("temperature", &motifcl::nn::GenerateOptions::temperature)
        .def_readwrite("top_k", &motifcl::nn::GenerateOptions::top_k)
        .def_readwrite("top_p", &motifcl::nn::GenerateOptions::top_p)
        .def_readwrite("bos_token_id", &motifcl::nn::GenerateOptions::bos_token_id)
        .def_readwrite("eos_token_id", &motifcl::nn::GenerateOptions::eos_token_id)
        .def_readwrite("pad_token_id", &motifcl::nn::GenerateOptions::pad_token_id)
        .def_readwrite("add_bos", &motifcl::nn::GenerateOptions::add_bos)
        .def_readwrite("prefill_prompt", &motifcl::nn::GenerateOptions::prefill_prompt)
        .def_readwrite("gpu_greedy_sampling", &motifcl::nn::GenerateOptions::gpu_greedy_sampling)
        .def_readwrite("use_paged_kv_cache", &motifcl::nn::GenerateOptions::use_paged_kv_cache)
        .def_readwrite("kv_page_size", &motifcl::nn::GenerateOptions::kv_page_size)
        .def_readwrite("seed", &motifcl::nn::GenerateOptions::seed);

    py::enum_<motifcl::nn::HFArchitecture>(m, "HFArchitecture")
        .value("Auto", motifcl::nn::HFArchitecture::Auto)
        .value("GenericDecoder", motifcl::nn::HFArchitecture::GenericDecoder)
        .value("Gemma", motifcl::nn::HFArchitecture::Gemma)
        .value("Gemma2", motifcl::nn::HFArchitecture::Gemma2)
        .value("Gemma3", motifcl::nn::HFArchitecture::Gemma3)
        .value("Gemma4", motifcl::nn::HFArchitecture::Gemma4)
        .value("Llama", motifcl::nn::HFArchitecture::Llama)
        .value("Mistral", motifcl::nn::HFArchitecture::Mistral)
        .value("Qwen2", motifcl::nn::HFArchitecture::Qwen2)
        .value("Qwen3", motifcl::nn::HFArchitecture::Qwen3)
        .value("Qwen35", motifcl::nn::HFArchitecture::Qwen35)
        .value("Phi3", motifcl::nn::HFArchitecture::Phi3)
        .value("Phi4", motifcl::nn::HFArchitecture::Phi4)
        .value("Mixtral", motifcl::nn::HFArchitecture::Mixtral)
        .value("DeepSeek", motifcl::nn::HFArchitecture::DeepSeek)
        .value("Falcon", motifcl::nn::HFArchitecture::Falcon)
        .value("GPTNeoX", motifcl::nn::HFArchitecture::GPTNeoX)
        .value("Mamba", motifcl::nn::HFArchitecture::Mamba);

    py::enum_<motifcl::nn::HFChatTemplateKind>(m, "HFChatTemplateKind")
        .value("Auto", motifcl::nn::HFChatTemplateKind::Auto)
        .value("None_", motifcl::nn::HFChatTemplateKind::None)
        .value("Generic", motifcl::nn::HFChatTemplateKind::Generic)
        .value("ChatML", motifcl::nn::HFChatTemplateKind::ChatML)
        .value("Llama2", motifcl::nn::HFChatTemplateKind::Llama2)
        .value("Llama3", motifcl::nn::HFChatTemplateKind::Llama3)
        .value("Mistral", motifcl::nn::HFChatTemplateKind::Mistral)
        .value("Gemma", motifcl::nn::HFChatTemplateKind::Gemma);

    py::enum_<motifcl::nn::ModernLayerKind>(m, "ModernLayerKind")
        .value("FullAttention", motifcl::nn::ModernLayerKind::FullAttention)
        .value("SlidingAttention", motifcl::nn::ModernLayerKind::SlidingAttention)
        .value("GatedDeltaNet", motifcl::nn::ModernLayerKind::GatedDeltaNet)
        .value("GatedAttention", motifcl::nn::ModernLayerKind::GatedAttention)
        .value("MoEFFN", motifcl::nn::ModernLayerKind::MoEFFN)
        .value("VisionProjector", motifcl::nn::ModernLayerKind::VisionProjector)
        .value("AudioProjector", motifcl::nn::ModernLayerKind::AudioProjector)
        .value("SwiGLUFFN", motifcl::nn::ModernLayerKind::SwiGLUFFN);

    py::class_<motifcl::nn::LayerSpec>(m, "LayerSpec")
        .def_readonly("graph_index", &motifcl::nn::LayerSpec::graph_index)
        .def_readonly("transformer_layer", &motifcl::nn::LayerSpec::transformer_layer)
        .def_readonly("kind", &motifcl::nn::LayerSpec::kind)
        .def_readonly("sliding_window", &motifcl::nn::LayerSpec::sliding_window)
        .def_readonly("num_experts", &motifcl::nn::LayerSpec::num_experts)
        .def_readonly("experts_per_token", &motifcl::nn::LayerSpec::experts_per_token)
        .def_readonly("uses_kv_cache", &motifcl::nn::LayerSpec::uses_kv_cache)
        .def_readonly("uses_state_cache", &motifcl::nn::LayerSpec::uses_state_cache)
        .def_readonly("consumes_per_layer_input", &motifcl::nn::LayerSpec::consumes_per_layer_input)
        .def_readonly("name", &motifcl::nn::LayerSpec::name);

    py::class_<motifcl::nn::ModernModelSpec>(m, "ModernModelSpec")
        .def_readonly("architecture", &motifcl::nn::ModernModelSpec::architecture)
        .def_readonly("architecture_name", &motifcl::nn::ModernModelSpec::architecture_name)
        .def_readonly("transformer", &motifcl::nn::ModernModelSpec::transformer)
        .def_readonly("layers", &motifcl::nn::ModernModelSpec::layers)
        .def_readonly("text_core", &motifcl::nn::ModernModelSpec::text_core)
        .def_readonly("per_layer_inputs", &motifcl::nn::ModernModelSpec::per_layer_inputs)
        .def_readonly("has_moe", &motifcl::nn::ModernModelSpec::has_moe)
        .def_readonly("has_vision_projector", &motifcl::nn::ModernModelSpec::has_vision_projector)
        .def_readonly("has_audio_projector", &motifcl::nn::ModernModelSpec::has_audio_projector)
        .def_readonly("has_recurrent_state", &motifcl::nn::ModernModelSpec::has_recurrent_state)
        .def_readonly("blockers", &motifcl::nn::ModernModelSpec::blockers);

    py::class_<motifcl::nn::LongContextRuntimeSpec>(m, "LongContextRuntimeSpec")
        .def_readonly("max_context", &motifcl::nn::LongContextRuntimeSpec::max_context)
        .def_readonly("page_size", &motifcl::nn::LongContextRuntimeSpec::page_size)
        .def_readonly("sliding_window", &motifcl::nn::LongContextRuntimeSpec::sliding_window)
        .def_readonly("kv_cache_layers", &motifcl::nn::LongContextRuntimeSpec::kv_cache_layers)
        .def_readonly("sliding_window_layers", &motifcl::nn::LongContextRuntimeSpec::sliding_window_layers)
        .def_readonly("state_cache_layers", &motifcl::nn::LongContextRuntimeSpec::state_cache_layers)
        .def_readonly("needs_paged_kv", &motifcl::nn::LongContextRuntimeSpec::needs_paged_kv)
        .def_readonly("needs_sliding_window_cache", &motifcl::nn::LongContextRuntimeSpec::needs_sliding_window_cache)
        .def_readonly("needs_state_cache", &motifcl::nn::LongContextRuntimeSpec::needs_state_cache);

    py::class_<motifcl::nn::HFTransformerConfig>(m, "HFTransformerConfig")
        .def(py::init<>())
        .def_readwrite("architecture", &motifcl::nn::HFTransformerConfig::architecture)
        .def_readwrite("architecture_name", &motifcl::nn::HFTransformerConfig::architecture_name)
        .def_readwrite("transformer", &motifcl::nn::HFTransformerConfig::transformer)
        .def_readwrite("rms_norm_eps", &motifcl::nn::HFTransformerConfig::rms_norm_eps)
        .def_readwrite("tie_word_embeddings", &motifcl::nn::HFTransformerConfig::tie_word_embeddings)
        .def_readwrite("attention_bias", &motifcl::nn::HFTransformerConfig::attention_bias)
        .def_readwrite("attention_k_eq_v", &motifcl::nn::HFTransformerConfig::attention_k_eq_v)
        .def_readwrite("bos_token_id", &motifcl::nn::HFTransformerConfig::bos_token_id)
        .def_readwrite("eos_token_id", &motifcl::nn::HFTransformerConfig::eos_token_id)
        .def_readwrite("pad_token_id", &motifcl::nn::HFTransformerConfig::pad_token_id)
        .def_readwrite("sliding_window", &motifcl::nn::HFTransformerConfig::sliding_window)
        .def_readwrite("layer_types", &motifcl::nn::HFTransformerConfig::layer_types)
        .def_readwrite("local_attention_layers", &motifcl::nn::HFTransformerConfig::local_attention_layers)
        .def_readwrite("global_attention_layers", &motifcl::nn::HFTransformerConfig::global_attention_layers)
        .def_readwrite("per_layer_inputs", &motifcl::nn::HFTransformerConfig::per_layer_inputs)
        .def_readwrite("has_moe", &motifcl::nn::HFTransformerConfig::has_moe)
        .def_readwrite("num_experts", &motifcl::nn::HFTransformerConfig::num_experts)
        .def_readwrite("experts_per_token", &motifcl::nn::HFTransformerConfig::experts_per_token)
        .def_readwrite("has_vision_projector", &motifcl::nn::HFTransformerConfig::has_vision_projector)
        .def_readwrite("has_audio_projector", &motifcl::nn::HFTransformerConfig::has_audio_projector)
        .def_readwrite("has_gated_delta_net", &motifcl::nn::HFTransformerConfig::has_gated_delta_net)
        .def_readwrite("has_gated_attention", &motifcl::nn::HFTransformerConfig::has_gated_attention);

    py::class_<motifcl::nn::HFChatMessage>(m, "HFChatMessage")
        .def(py::init<>())
        .def(py::init([](std::string role, std::string content) {
            motifcl::nn::HFChatMessage msg;
            msg.role = std::move(role);
            msg.content = std::move(content);
            return msg;
        }), py::arg("role"), py::arg("content"))
        .def_readwrite("role", &motifcl::nn::HFChatMessage::role)
        .def_readwrite("content", &motifcl::nn::HFChatMessage::content);

    m.attr("HFWeightName") = m.attr("GemmaWeightName");
    m.attr("HFWeightLoadReport") = m.attr("GemmaWeightLoadReport");
    m.attr("HFTokenizer") = m.attr("GemmaTokenizer");

    m.def("load_gemma_config_json", &motifcl::nn::load_gemma_config_json, py::arg("path"));
    m.def("to_transformer_config", &motifcl::nn::to_transformer_config, py::arg("config"));
    m.def("make_gemma_model", &motifcl::nn::make_gemma_model, py::arg("backend"), py::arg("config"));
    m.def("map_gemma_hf_weight_name", &motifcl::nn::map_gemma_hf_weight_name, py::arg("hf_name"));
    m.def("expected_gemma_hf_weight_names", &motifcl::nn::expected_gemma_hf_weight_names,
          py::arg("config"), py::arg("include_lm_head") = true);
    m.def("load_gemma_hf_weights", &motifcl::nn::load_gemma_hf_weights,
          py::arg("backend"), py::arg("model"), py::arg("safetensors_paths"), py::arg("config"),
          py::arg("strict") = false, py::arg("trainable") = false);
    m.def("generate", &motifcl::nn::generate,
          py::arg("backend"), py::arg("model"), py::arg("prompt_tokens"),
          py::arg("options") = motifcl::nn::GenerateOptions{});
    m.def("generate_text", &motifcl::nn::generate_text,
          py::arg("backend"), py::arg("model"), py::arg("tokenizer"), py::arg("prompt"),
          py::arg("options") = motifcl::nn::GenerateOptions{});
    m.def("generate_batch", &motifcl::nn::generate_batch,
          py::arg("backend"), py::arg("model"), py::arg("prompt_tokens"),
          py::arg("options") = motifcl::nn::GenerateOptions{});
    m.def("generate_batch_text", &motifcl::nn::generate_batch_text,
          py::arg("backend"), py::arg("model"), py::arg("tokenizer"), py::arg("prompts"),
          py::arg("options") = motifcl::nn::GenerateOptions{});
    m.def("hf_architecture_name", &motifcl::nn::hf_architecture_name, py::arg("architecture"));
    m.def("parse_hf_architecture", &motifcl::nn::parse_hf_architecture, py::arg("value"));
    m.def("modern_layer_kind_name", &motifcl::nn::modern_layer_kind_name, py::arg("kind"));
    m.def("parse_modern_layer_kind", &motifcl::nn::parse_modern_layer_kind, py::arg("value"));
    m.def("load_hf_transformer_config_json", &motifcl::nn::load_hf_transformer_config_json,
          py::arg("path"), py::arg("architecture") = motifcl::nn::HFArchitecture::Auto);
    m.def("load_hf_transformer_config_gguf", &motifcl::nn::load_hf_transformer_config_gguf,
          py::arg("path"), py::arg("architecture") = motifcl::nn::HFArchitecture::Auto);
    m.def("modern_model_spec_from_config", &motifcl::nn::modern_model_spec_from_config, py::arg("config"));
    m.def("load_modern_model_spec_json", &motifcl::nn::load_modern_model_spec_json,
          py::arg("path"), py::arg("architecture") = motifcl::nn::HFArchitecture::Auto);
    m.def("modern_model_spec_runnable_by_modern_gpt", &motifcl::nn::modern_model_spec_runnable_by_modern_gpt,
          py::arg("spec"));
    m.def("modern_model_spec_runnable_by_hybrid", &motifcl::nn::modern_model_spec_runnable_by_hybrid,
          py::arg("spec"));
    m.def("modern_model_spec_blockers", &motifcl::nn::modern_model_spec_blockers, py::arg("spec"));
    m.def("long_context_runtime_spec_from_model_spec", &motifcl::nn::long_context_runtime_spec_from_model_spec,
          py::arg("spec"), py::arg("page_size") = 256);
    m.def("format_modern_model_spec", &motifcl::nn::format_modern_model_spec, py::arg("spec"));
    m.def("infer_hf_chat_template_kind", &motifcl::nn::infer_hf_chat_template_kind,
          py::arg("architecture"), py::arg("model_dir_or_tokenizer_config") = "");
    m.def("apply_hf_chat_template", &motifcl::nn::apply_hf_chat_template,
          py::arg("messages"),
          py::arg("architecture") = motifcl::nn::HFArchitecture::GenericDecoder,
          py::arg("kind") = motifcl::nn::HFChatTemplateKind::Auto,
          py::arg("add_generation_prompt") = true);
    m.def("to_gemma_compatible_config", &motifcl::nn::to_gemma_compatible_config, py::arg("config"));
    m.def("make_hf_transformer_model", &motifcl::nn::make_hf_transformer_model,
          py::arg("backend"), py::arg("config"));
    m.def("hybrid_layer_configs_from_model_spec", &motifcl::nn::hybrid_layer_configs_from_model_spec,
          py::arg("spec"));
    m.def("make_hf_hybrid_transformer_model", &motifcl::nn::make_hf_hybrid_transformer_model,
          py::arg("backend"), py::arg("config"));
    m.def("map_hf_transformer_weight_name", &motifcl::nn::map_hf_transformer_weight_name,
          py::arg("architecture"), py::arg("hf_name"));
    m.def("expected_hf_transformer_weight_names", &motifcl::nn::expected_hf_transformer_weight_names,
          py::arg("config"), py::arg("include_lm_head") = true);
    m.def("load_hf_transformer_weights", &motifcl::nn::load_hf_transformer_weights,
          py::arg("backend"), py::arg("model"), py::arg("safetensors_paths"), py::arg("config"),
          py::arg("strict") = false, py::arg("trainable") = false);
    m.def("load_hf_transformer_gguf_weights", &motifcl::nn::load_hf_transformer_gguf_weights,
          py::arg("backend"), py::arg("model"), py::arg("gguf_path"), py::arg("config"),
          py::arg("strict") = false, py::arg("trainable") = false);
    m.def("load_hf_hybrid_transformer_weights", &motifcl::nn::load_hf_hybrid_transformer_weights,
          py::arg("backend"), py::arg("model"), py::arg("safetensors_paths"), py::arg("config"),
          py::arg("strict") = false, py::arg("trainable") = false);
    m.def("enable_hf_transformer_quantized_inference", &motifcl::nn::enable_hf_transformer_quantized_inference,
          py::arg("model"), py::arg("qdtype") = motifcl::DType::Q4_0);
    m.def("disable_hf_transformer_quantized_inference", &motifcl::nn::disable_hf_transformer_quantized_inference,
          py::arg("model"));
    m.def("load_hf_tokenizer", &motifcl::nn::load_hf_tokenizer,
          py::arg("model_dir_or_vocab_path"), py::arg("config"));
    m.def("generate_hf_text", &motifcl::nn::generate_hf_text,
          py::arg("backend"), py::arg("model"), py::arg("tokenizer"), py::arg("prompt"),
          py::arg("options") = motifcl::nn::GenerateOptions{});
    m.def("generate_hf_hybrid_text", &motifcl::nn::generate_hf_hybrid_text,
          py::arg("backend"), py::arg("model"), py::arg("tokenizer"), py::arg("prompt"),
          py::arg("options") = motifcl::nn::GenerateOptions{});
    m.def("generate_hf_batch_text", &motifcl::nn::generate_hf_batch_text,
          py::arg("backend"), py::arg("model"), py::arg("tokenizer"), py::arg("prompts"),
          py::arg("options") = motifcl::nn::GenerateOptions{});

    py::class_<motifcl::optim::Adam>(m, "Adam")
        .def(py::init<std::vector<motifcl::nn::Parameter*>, float, float, float, float, float>(), py::arg("params"), py::arg("lr") = 1e-3f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
        .def("step", &motifcl::optim::Adam::step)
        .def("zero_grad", &motifcl::optim::Adam::zero_grad)
        .def("set_lr", &motifcl::optim::Adam::set_lr)
        .def("lr", &motifcl::optim::Adam::lr)
        .def("set_weight_decay", &motifcl::optim::Adam::set_weight_decay)
        .def("weight_decay", &motifcl::optim::Adam::weight_decay);

    py::class_<motifcl::optim::LossScaleUpdate>(m, "LossScaleUpdate")
        .def_readonly("found_inf", &motifcl::optim::LossScaleUpdate::found_inf)
        .def_readonly("scale", &motifcl::optim::LossScaleUpdate::scale);

    py::class_<motifcl::optim::DynamicLossScaler>(m, "DynamicLossScaler")
        .def(py::init<float, float, float, int>(),
             py::arg("initial_scale") = 65536.0f, py::arg("growth_factor") = 2.0f,
             py::arg("backoff_factor") = 0.5f, py::arg("growth_interval") = 2000)
        .def("scale_loss", &motifcl::optim::DynamicLossScaler::scale_loss)
        .def("unscale_", &motifcl::optim::DynamicLossScaler::unscale_)
        .def("has_overflow", &motifcl::optim::DynamicLossScaler::has_overflow)
        .def("update", &motifcl::optim::DynamicLossScaler::update)
        .def("scale", &motifcl::optim::DynamicLossScaler::scale)
        .def("set_scale", &motifcl::optim::DynamicLossScaler::set_scale)
        .def("growth_tracker", &motifcl::optim::DynamicLossScaler::growth_tracker);

    py::class_<motifcl::optim::MixedPrecisionAdam>(m, "MixedPrecisionAdam")
        .def(py::init<std::vector<motifcl::nn::Parameter*>, float, float, float, float, float>(),
             py::arg("params"), py::arg("lr") = 1e-3f, py::arg("beta1") = 0.9f,
             py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
        .def("step", &motifcl::optim::MixedPrecisionAdam::step)
        .def("step_scaled", &motifcl::optim::MixedPrecisionAdam::step_scaled)
        .def("zero_grad", &motifcl::optim::MixedPrecisionAdam::zero_grad)
        .def("set_lr", &motifcl::optim::MixedPrecisionAdam::set_lr)
        .def("lr", &motifcl::optim::MixedPrecisionAdam::lr)
        .def("set_weight_decay", &motifcl::optim::MixedPrecisionAdam::set_weight_decay)
        .def("weight_decay", &motifcl::optim::MixedPrecisionAdam::weight_decay)
        .def("step_count", &motifcl::optim::MixedPrecisionAdam::step_count)
        .def("refresh_master_from_params", &motifcl::optim::MixedPrecisionAdam::refresh_master_from_params);

    py::class_<motifcl::optim::SGD>(m, "SGD")
        .def(py::init<std::vector<motifcl::nn::Parameter*>, float>(), py::arg("params"), py::arg("lr") = 1e-2f)
        .def("step", &motifcl::optim::SGD::step)
        .def("zero_grad", &motifcl::optim::SGD::zero_grad)
        .def("set_lr", &motifcl::optim::SGD::set_lr)
        .def("lr", &motifcl::optim::SGD::lr);
}
