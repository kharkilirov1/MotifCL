#include <motifcl/autograd/graph.hpp>

#include <motifcl/core/error.hpp>
#include <motifcl/runtime/command_buffer.hpp>
#include <motifcl/runtime/opencl_context.hpp>
#include <motifcl/tensor/tensor.hpp>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace motifcl::autograd {

namespace {
thread_local bool g_graph_capture_active = false;
thread_local CapturedGraph g_graph_capture;

struct PendingReplayLaunch {
    std::vector<std::pair<int, cl_mem>> buffer_args;
    GraphKernelLaunchInfo command;
    bool has_command = false;
    std::function<void(const std::unordered_map<int, cl_mem>&,
                       const std::vector<std::pair<int, int>>&)>
        replay;
};

thread_local std::vector<PendingReplayLaunch> g_pending_replay_launches;

std::mutex g_tensor_registry_mutex;
std::unordered_map<int, TensorSpec> g_tensor_registry;

std::vector<int> concat_ids(const std::vector<int>& a, const std::vector<int>& b, const std::vector<int>& c = {}) {
    std::vector<int> ids;
    ids.reserve(a.size() + b.size() + c.size());
    ids.insert(ids.end(), a.begin(), a.end());
    ids.insert(ids.end(), b.begin(), b.end());
    ids.insert(ids.end(), c.begin(), c.end());
    return ids;
}

void retain_spec_mem(TensorSpec& spec) {
    if (!spec.mem || spec.retained_mem) return;
    MCL_CHECK_CL(clRetainMemObject(spec.mem));
    using MemHandle = std::remove_pointer_t<cl_mem>;
    spec.retained_mem = std::shared_ptr<MemHandle>(spec.mem, [](MemHandle* mem) {
        if (mem) clReleaseMemObject(mem);
    });
}

int resolve_alias(const std::unordered_map<int, int>& aliases, int id) {
    std::unordered_set<int> seen;
    auto it = aliases.find(id);
    while (it != aliases.end() && seen.insert(id).second) {
        id = it->second;
        it = aliases.find(id);
    }
    return id;
}

bool allocates_outputs(const GraphNodeInfo& node) {
    return node.op != "view";
}

bool graph_temporary_arena_enabled() {
    const char* env = std::getenv("MOTIFCL_DISABLE_GRAPH_TEMP_ARENA");
    return !(env && *env && std::string(env) != "0" && std::string(env) != "false" && std::string(env) != "FALSE");
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

bool graph_is_driver_command_buffer_recordable(const CapturedGraph& graph, const std::vector<std::size_t>& schedule) {
    if (schedule.empty()) return false;
    bool has_kernel_launch = false;
    for (std::size_t index : schedule) {
        const auto& node = graph.nodes().at(index);
        if (!node.replayable || !node.command_buffer_recordable) return false;
        has_kernel_launch = has_kernel_launch || !node.kernel_launches().empty();
    }
    return has_kernel_launch;
}

std::vector<GraphKernelLaunchInfo> flatten_driver_kernel_launches(const CapturedGraph& graph,
                                                                  const std::vector<std::size_t>& schedule) {
    std::vector<GraphKernelLaunchInfo> launches;
    for (std::size_t index : schedule) {
        const auto& node_launches = graph.nodes().at(index).kernel_launches();
        launches.insert(launches.end(), node_launches.begin(), node_launches.end());
    }
    return launches;
}

std::vector<std::pair<int, int>> map_kernel_buffer_args_to_tensors(const PendingReplayLaunch& launch,
                                                                   const std::vector<int>& tensor_ids,
                                                                   const std::unordered_map<int, TensorSpec>& specs) {
    std::vector<std::pair<int, int>> result;
    result.reserve(launch.buffer_args.size());
    for (std::size_t i = 0; i < launch.buffer_args.size(); ++i) {
        const auto& arg = launch.buffer_args[i];
        bool mapped = false;
        if (arg.second != nullptr) {
            for (int tensor_id : tensor_ids) {
                auto spec = specs.find(tensor_id);
                if (spec != specs.end() && spec->second.mem == arg.second) {
                    result.push_back({arg.first, tensor_id});
                    mapped = true;
                    break;
                }
            }
        }
        if (!mapped && i < tensor_ids.size()) {
            result.push_back({arg.first, tensor_ids[i]});
        }
    }
    return result;
}
}

void GraphNodeInfo::replay() const {
    replay({});
}

void GraphNodeInfo::replay(const std::unordered_map<int, cl_mem>& tensor_bindings) const {
    MCL_CHECK(replayable && replay_fn, "graph node is not replayable: " + op);
    replay_fn(tensor_bindings);
}

bool CapturedGraph::replayable() const {
    return std::all_of(nodes_.begin(), nodes_.end(), [](const GraphNodeInfo& node) {
        return node.replayable;
    });
}

bool CapturedGraph::rebindable() const {
    return replayable() && std::any_of(nodes_.begin(), nodes_.end(), [](const GraphNodeInfo& node) {
        return node.rebindable;
    });
}

std::vector<std::size_t> CapturedGraph::schedule() const {
    std::vector<std::size_t> order;
    order.reserve(nodes_.size());
    std::unordered_set<int> produced;
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        order.push_back(i);
        for (int id : nodes_[i].outputs) produced.insert(id);
    }
    return order;
}

std::vector<TensorSpec> CapturedGraph::tensor_specs_list() const {
    std::vector<TensorSpec> specs;
    specs.reserve(tensor_specs_.size());
    for (const auto& item : tensor_specs_) specs.push_back(item.second);
    std::sort(specs.begin(), specs.end(), [](const TensorSpec& a, const TensorSpec& b) {
        return a.id < b.id;
    });
    return specs;
}

GraphBufferPlan CapturedGraph::plan_buffers(const GraphOptimizeOptions& options) const {
    struct Interval {
        int tensor_id = 0;
        std::size_t bytes = 0;
        std::size_t first = 0;
        std::size_t last = 0;
        std::size_t allocation_id = 0;
    };

    GraphBufferPlan plan;
    plan.shape_polymorphic = options.shape_polymorphic;
    if (nodes_.empty()) return plan;

    std::unordered_map<int, int> aliases;
    std::unordered_map<int, std::size_t> first;
    std::unordered_map<int, std::size_t> last;
    std::unordered_map<int, std::size_t> bytes;

    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        const auto& node = nodes_[i];
        for (int input : node.inputs) {
            const int root = resolve_alias(aliases, input);
            auto found = first.find(root);
            if (found != first.end()) last[root] = std::max(last[root], i);
        }

        if (node.op == "view" && !node.inputs.empty()) {
            const int root = resolve_alias(aliases, node.inputs.front());
            for (int output : node.outputs) aliases[output] = root;
        } else if (allocates_outputs(node)) {
            for (int output : node.outputs) {
                const auto spec = tensor_specs_.find(output);
                if (spec == tensor_specs_.end()) continue;
                if (first.find(output) == first.end()) {
                    first[output] = i;
                    last[output] = i;
                    bytes[output] = spec->second.nbytes;
                }
                plan.total_output_bytes += spec->second.nbytes;
                plan.shape_polymorphic = plan.shape_polymorphic || spec->second.dynamic_shape;
            }
        }

        for (int temporary : node.temporaries) {
            const auto spec = tensor_specs_.find(temporary);
            if (spec == tensor_specs_.end()) continue;
            if (first.find(temporary) == first.end()) {
                first[temporary] = i;
                last[temporary] = i;
                bytes[temporary] = spec->second.nbytes;
            }
            plan.total_output_bytes += spec->second.nbytes;
            plan.shape_polymorphic = plan.shape_polymorphic || spec->second.dynamic_shape;
        }
    }

    std::vector<Interval> intervals;
    intervals.reserve(first.size());
    for (const auto& item : first) {
        const int id = item.first;
        const auto byte_it = bytes.find(id);
        if (byte_it == bytes.end() || byte_it->second == 0) continue;
        intervals.push_back({id, byte_it->second, item.second, last[id], 0});
    }
    std::sort(intervals.begin(), intervals.end(), [](const Interval& a, const Interval& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.tensor_id < b.tensor_id;
    });

    std::vector<std::size_t> free_after;
    for (auto& interval : intervals) {
        std::size_t chosen = std::numeric_limits<std::size_t>::max();
        std::size_t best_bytes = std::numeric_limits<std::size_t>::max();
        if (options.enable_buffer_reuse) {
            for (std::size_t i = 0; i < plan.allocations.size(); ++i) {
                if (free_after[i] < interval.first &&
                    plan.allocations[i].bytes >= interval.bytes &&
                    plan.allocations[i].bytes < best_bytes) {
                    chosen = i;
                    best_bytes = plan.allocations[i].bytes;
                }
            }
        }
        if (chosen == std::numeric_limits<std::size_t>::max()) {
            chosen = plan.allocations.size();
            GraphBufferAllocation allocation;
            allocation.allocation_id = chosen;
            allocation.bytes = interval.bytes;
            allocation.first_node = interval.first;
            allocation.last_node = interval.last;
            allocation.tensor_ids.push_back(interval.tensor_id);
            plan.allocations.push_back(std::move(allocation));
            free_after.push_back(interval.last);
        } else {
            auto& allocation = plan.allocations[chosen];
            allocation.tensor_ids.push_back(interval.tensor_id);
            allocation.first_node = std::min(allocation.first_node, interval.first);
            allocation.last_node = std::max(allocation.last_node, interval.last);
            free_after[chosen] = interval.last;
        }
        interval.allocation_id = chosen;
    }

    for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
        std::unordered_set<std::size_t> active_allocations;
        for (const auto& interval : intervals) {
            if (interval.first <= node_index && node_index <= interval.last) {
                active_allocations.insert(interval.allocation_id);
            }
        }
        std::size_t active_bytes = 0;
        for (std::size_t allocation_id : active_allocations) {
            active_bytes += plan.allocations[allocation_id].bytes;
        }
        plan.peak_bytes = std::max(plan.peak_bytes, active_bytes);
    }

    std::size_t allocated_bytes = 0;
    for (const auto& allocation : plan.allocations) allocated_bytes += allocation.bytes;
    if (plan.total_output_bytes > allocated_bytes) {
        plan.reused_bytes = plan.total_output_bytes - allocated_bytes;
    }
    return plan;
}

GraphBufferPlan CapturedGraph::optimize(const GraphOptimizeOptions& options) const {
    return plan_buffers(options);
}

GraphRuntimePlan CapturedGraph::compile_runtime_plan(const std::vector<TensorSpec>& runtime_specs,
                                                     const GraphOptimizeOptions& options) const {
    GraphRuntimePlan result;
    result.runtime_specs = runtime_specs;
    result.compatible = compatible_with(runtime_specs, true);
    MCL_CHECK(result.compatible, "runtime tensor specs are incompatible with captured graph");

    CapturedGraph runtime_graph = *this;
    for (const auto& spec : runtime_specs) {
        auto it = runtime_graph.tensor_specs_.find(spec.id);
        if (it != runtime_graph.tensor_specs_.end()) {
            it->second.shape = spec.shape;
            it->second.dtype = spec.dtype;
            it->second.nbytes = spec.nbytes;
        }
    }
    result.buffer_plan = runtime_graph.plan_buffers(options);

    std::unordered_map<int, std::size_t> tensor_to_allocation;
    for (const auto& allocation : result.buffer_plan.allocations) {
        for (int tensor_id : allocation.tensor_ids) tensor_to_allocation[tensor_id] = allocation.allocation_id;
    }
    for (const auto& item : runtime_graph.tensor_specs_) {
        const auto alloc_it = tensor_to_allocation.find(item.first);
        if (alloc_it == tensor_to_allocation.end()) continue;
        GraphRuntimeBinding binding;
        binding.tensor_id = item.first;
        binding.allocation_id = alloc_it->second;
        binding.offset = 0;
        binding.nbytes = item.second.nbytes;
        binding.shape = item.second.shape;
        binding.dtype = item.second.dtype;
        result.bindings.push_back(std::move(binding));
    }
    std::sort(result.bindings.begin(), result.bindings.end(), [](const GraphRuntimeBinding& a, const GraphRuntimeBinding& b) {
        return a.tensor_id < b.tensor_id;
    });
    result.kernel_rebinding = rebindable();
    result.note = result.kernel_rebinding
        ? "runtime plan supports same-shape cl_mem rebinding for captured kernel buffer args through GraphExecutor::bind_tensor"
        : "runtime plan materializes tensor-to-allocation bindings; no captured kernel buffer args are safely rebindable";
    return result;
}

CapturedGraph CapturedGraph::shape_polymorphic() const {
    CapturedGraph result = *this;
    for (auto& item : result.tensor_specs_) {
        item.second.dynamic_shape = true;
        for (auto& dim : item.second.shape.dims) dim = -1;
    }
    return result;
}

bool CapturedGraph::compatible_with(const std::vector<TensorSpec>& runtime_specs, bool allow_dynamic_dims) const {
    std::unordered_map<int, TensorSpec> runtime_by_id;
    for (const auto& spec : runtime_specs) runtime_by_id[spec.id] = spec;

    for (const auto& item : tensor_specs_) {
        const auto& captured = item.second;
        const auto runtime_it = runtime_by_id.find(item.first);
        if (runtime_it == runtime_by_id.end()) return false;
        const auto& runtime = runtime_it->second;
        if (captured.dtype != runtime.dtype) return false;
        if (captured.shape.ndim() != runtime.shape.ndim()) return false;
        const bool dynamic = allow_dynamic_dims && captured.dynamic_shape;
        if (!dynamic) {
            if (captured.shape != runtime.shape) return false;
            if (captured.nbytes != runtime.nbytes) return false;
        } else {
            for (std::size_t i = 0; i < captured.shape.dims.size(); ++i) {
                if (captured.shape.dims[i] >= 0 && captured.shape.dims[i] != runtime.shape.dims[i]) {
                    return false;
                }
            }
        }
    }
    return true;
}

void CapturedGraph::replay() const {
    replay({});
}

void CapturedGraph::replay(const std::unordered_map<int, cl_mem>& tensor_bindings) const {
    MCL_CHECK(replayable(), "captured graph contains nodes that cannot be replayed");
    for (std::size_t index : schedule()) {
        nodes_.at(index).replay(tensor_bindings);
    }
}

void CapturedGraph::remember_tensor_specs(const std::vector<int>& ids) {
    std::lock_guard<std::mutex> lock(g_tensor_registry_mutex);
    for (int id : ids) {
        auto it = g_tensor_registry.find(id);
        if (it != g_tensor_registry.end()) {
            TensorSpec spec = it->second;
            auto existing = tensor_specs_.find(id);
            if (existing != tensor_specs_.end() && existing->second.mem == spec.mem) {
                spec.retained_mem = existing->second.retained_mem;
            }
            retain_spec_mem(spec);
            tensor_specs_[id] = std::move(spec);
        }
    }
}

void begin_graph_capture() {
    MCL_CHECK(!g_graph_capture_active, "graph capture is already active");
    g_graph_capture.clear();
    g_pending_replay_launches.clear();
    g_graph_capture_active = true;
}

CapturedGraph end_graph_capture() {
    MCL_CHECK(g_graph_capture_active, "graph capture is not active");
    g_graph_capture_active = false;
    CapturedGraph result = g_graph_capture;
    g_graph_capture.clear();
    g_pending_replay_launches.clear();
    return result;
}

bool is_graph_capturing() { return g_graph_capture_active; }

const CapturedGraph& current_graph_capture() {
    MCL_CHECK(g_graph_capture_active, "graph capture is not active");
    return g_graph_capture;
}

void clear_graph_capture() {
    g_graph_capture.clear();
    g_pending_replay_launches.clear();
    g_graph_capture_active = false;
}

void record_op(const std::string& op, std::vector<int> inputs, std::vector<int> outputs) {
    record_op(op, std::move(inputs), std::move(outputs), !g_pending_replay_launches.empty());
}

void record_op(const std::string& op, std::vector<int> inputs, std::vector<int> outputs, bool replayable) {
    if (!g_graph_capture_active) return;
    GraphNodeInfo node;
    node.op = op;
    node.inputs = std::move(inputs);
    node.outputs = std::move(outputs);
    node.replayable = replayable;
    if (replayable) {
        auto launches = std::move(g_pending_replay_launches);
        const auto tensor_ids = concat_ids(node.inputs, node.outputs);
        std::unordered_map<int, TensorSpec> specs_snapshot;
        {
            std::lock_guard<std::mutex> lock(g_tensor_registry_mutex);
            specs_snapshot = g_tensor_registry;
        }
        std::vector<std::vector<std::pair<int, int>>> arg_maps;
        arg_maps.reserve(launches.size());
        bool all_driver_recordable = !launches.empty();
        for (const auto& launch : launches) {
            arg_maps.push_back(map_kernel_buffer_args_to_tensors(launch, tensor_ids, specs_snapshot));
            node.rebindable = node.rebindable || !arg_maps.back().empty();
            all_driver_recordable = all_driver_recordable && launch.has_command;
            if (launch.has_command) {
                auto command = launch.command;
                command.tensor_arg_bindings = arg_maps.back();
                node.kernel_launches_.push_back(std::move(command));
            }
        }
        node.command_buffer_recordable = all_driver_recordable || (node.op == "view" && launches.empty());
        node.replay_fn = [launches = std::move(launches), arg_maps = std::move(arg_maps)](const std::unordered_map<int, cl_mem>& bindings) {
            for (std::size_t i = 0; i < launches.size(); ++i) launches[i].replay(bindings, arg_maps[i]);
        };
    }
    g_pending_replay_launches.clear();
    g_graph_capture.remember_tensor_specs(concat_ids(node.inputs, node.outputs));
    g_graph_capture.nodes().push_back(std::move(node));
}

void record_replay_op(const std::string& op,
                      std::vector<int> inputs,
                      std::vector<int> outputs,
                      std::function<void()> replay) {
    record_replay_op(op, std::move(inputs), std::move(outputs), {}, std::move(replay));
}

void record_replay_op(const std::string& op,
                      std::vector<int> inputs,
                      std::vector<int> outputs,
                      std::vector<int> temporaries,
                      std::function<void()> replay) {
    if (!g_graph_capture_active) return;
    GraphNodeInfo node;
    node.op = op;
    node.inputs = std::move(inputs);
    node.outputs = std::move(outputs);
    node.temporaries = std::move(temporaries);
    node.replayable = static_cast<bool>(replay);
    if (node.replayable) {
        auto launches = std::move(g_pending_replay_launches);
        const auto tensor_ids = concat_ids(node.inputs, node.outputs, node.temporaries);
        std::unordered_map<int, TensorSpec> specs_snapshot;
        {
            std::lock_guard<std::mutex> lock(g_tensor_registry_mutex);
            specs_snapshot = g_tensor_registry;
        }
        std::vector<std::vector<std::pair<int, int>>> arg_maps;
        arg_maps.reserve(launches.size());
        bool all_driver_recordable = !launches.empty();
        for (const auto& launch : launches) {
            arg_maps.push_back(map_kernel_buffer_args_to_tensors(launch, tensor_ids, specs_snapshot));
            node.rebindable = node.rebindable || !arg_maps.back().empty();
            all_driver_recordable = all_driver_recordable && launch.has_command;
            if (launch.has_command) {
                auto command = launch.command;
                command.tensor_arg_bindings = arg_maps.back();
                node.kernel_launches_.push_back(std::move(command));
            }
        }
        node.command_buffer_recordable = all_driver_recordable;
        node.replay_fn = [launches = std::move(launches), arg_maps = std::move(arg_maps), replay = std::move(replay)](const std::unordered_map<int, cl_mem>& bindings) {
            for (std::size_t i = 0; i < launches.size(); ++i) launches[i].replay(bindings, arg_maps[i]);
            replay();
        };
    }
    g_pending_replay_launches.clear();
    g_graph_capture.remember_tensor_specs(concat_ids(node.inputs, node.outputs, node.temporaries));
    g_graph_capture.nodes().push_back(std::move(node));
}

void record_kernel_launch(const std::string& kernel_name, std::function<void()> replay) {
    (void)kernel_name;
    if (!g_graph_capture_active) return;
    PendingReplayLaunch launch;
    launch.replay = [replay = std::move(replay)](const std::unordered_map<int, cl_mem>&,
                                                 const std::vector<std::pair<int, int>>&) {
        replay();
    };
    g_pending_replay_launches.push_back(std::move(launch));
}

void record_kernel_launch(const std::string& kernel_name,
                          std::vector<std::pair<int, cl_mem>> buffer_args,
                          std::function<void(const std::unordered_map<int, cl_mem>&,
                                             const std::vector<std::pair<int, int>>&)>
                              replay) {
    (void)kernel_name;
    if (!g_graph_capture_active) return;
    PendingReplayLaunch launch;
    launch.buffer_args = std::move(buffer_args);
    launch.replay = std::move(replay);
    g_pending_replay_launches.push_back(std::move(launch));
}

void record_kernel_launch(GraphKernelLaunchInfo launch,
                          std::function<void(const std::unordered_map<int, cl_mem>&,
                                             const std::vector<std::pair<int, int>>&)>
                              replay) {
    if (!g_graph_capture_active) return;
    PendingReplayLaunch pending;
    pending.buffer_args = launch.buffer_args;
    pending.command = std::move(launch);
    pending.has_command = pending.command.kernel != nullptr &&
                          pending.command.queue != nullptr &&
                          pending.command.work_dim > 0 &&
                          !pending.command.global_work_size.empty();
    pending.replay = std::move(replay);
    g_pending_replay_launches.push_back(std::move(pending));
}

void register_tensor(int id, const Shape& shape, DType dtype, std::size_t nbytes, cl_mem mem) {
    TensorSpec spec;
    spec.id = id;
    spec.shape = shape;
    spec.dtype = dtype;
    spec.nbytes = nbytes;
    spec.mem = mem;
    std::lock_guard<std::mutex> lock(g_tensor_registry_mutex);
    g_tensor_registry[id] = std::move(spec);
}

GraphCaptureGuard::GraphCaptureGuard() : active_(true) {
    begin_graph_capture();
}

GraphCaptureGuard::~GraphCaptureGuard() {
    if (active_) clear_graph_capture();
}

CapturedGraph GraphCaptureGuard::finish() {
    MCL_CHECK(active_, "graph capture guard already finished");
    active_ = false;
    return end_graph_capture();
}

GraphExecutor::GraphExecutor(CapturedGraph graph, const GraphOptimizeOptions& options)
    : graph_(std::move(graph)), options_(options) {
    runtime_plan_ = graph_.compile_runtime_plan(graph_.tensor_specs_list(), options_);
    cached_schedule_ = graph_.schedule();
    initialize_temporary_arena();
    if (graph_is_driver_command_buffer_recordable(graph_, cached_schedule_)) {
        auto launches = flatten_driver_kernel_launches(graph_, cached_schedule_);
        driver_command_buffer_ = ::motifcl::DriverCommandBuffer::try_create(launches);
        if (driver_command_buffer_ && driver_command_buffer_->valid()) {
            execution_mode_ = driver_command_buffer_->mode();
        }
    }
}

GraphExecutor::~GraphExecutor() = default;
GraphExecutor::GraphExecutor(GraphExecutor&&) noexcept = default;
GraphExecutor& GraphExecutor::operator=(GraphExecutor&&) noexcept = default;

void GraphExecutor::execute() {
    MCL_CHECK(graph_.replayable(), "captured graph contains nodes that cannot be replayed");
    std::unordered_map<int, cl_mem> active_bindings = arena_mem_;
    for (const auto& binding : bound_mem_) active_bindings[binding.first] = binding.second;
    if (driver_command_buffer_ && driver_command_buffer_->can_execute_with_bindings(active_bindings) &&
        driver_command_buffer_->execute(active_bindings)) {
        execution_mode_ = driver_command_buffer_->mode();
        ++executions_;
        return;
    }
    execution_mode_ = "host_replay_fallback";
    for (std::size_t index : cached_schedule_) {
        graph_.nodes().at(index).replay(active_bindings);
    }
    ++executions_;
}

void GraphExecutor::bind_tensor(int captured_tensor_id, const Tensor& tensor) {
    MCL_CHECK(tensor.valid(), "cannot bind invalid tensor to graph executor");
    const auto it = graph_.tensor_specs().find(captured_tensor_id);
    MCL_CHECK(it != graph_.tensor_specs().end(), "captured tensor id is not present in graph");
    const auto& spec = it->second;
    MCL_CHECK(spec.dtype == tensor.dtype(), "graph tensor binding dtype mismatch");
    MCL_CHECK(spec.shape == tensor.shape(), "graph tensor binding requires exact same shape");
    MCL_CHECK(spec.nbytes == tensor.nbytes(), "graph tensor binding byte-size mismatch");
    bound_tensors_[captured_tensor_id] = std::make_shared<Tensor>(tensor);
    bound_mem_[captured_tensor_id] = tensor.buffer().raw();
}

void GraphExecutor::clear_bindings() {
    bound_mem_.clear();
    bound_tensors_.clear();
}

void GraphExecutor::initialize_temporary_arena() {
    if (!graph_temporary_arena_enabled() || runtime_plan_.buffer_plan.allocations.empty()) return;

    std::unordered_set<int> temporary_ids;
    for (const auto& node : graph_.nodes()) {
        for (int id : node.temporaries) temporary_ids.insert(id);
    }
    if (temporary_ids.empty()) return;

    std::shared_ptr<OpenCLContextState> state;
    for (std::size_t index : cached_schedule_) {
        for (const auto& launch : graph_.nodes().at(index).kernel_launches()) {
            if (launch.retained_state && launch.retained_state->valid()) {
                state = launch.retained_state;
                break;
            }
        }
        if (state) break;
    }
    if (!state || !state->valid()) return;

    constexpr std::size_t kArenaAlignment = 256;
    struct ArenaAllocation {
        std::size_t allocation_id = 0;
        std::size_t bytes = 0;
        std::size_t offset = 0;
        std::vector<int> temporary_tensor_ids;
    };
    std::vector<ArenaAllocation> arena_allocations;
    std::size_t total_bytes = 0;
    for (const auto& allocation : runtime_plan_.buffer_plan.allocations) {
        ArenaAllocation arena_allocation;
        arena_allocation.allocation_id = allocation.allocation_id;
        arena_allocation.bytes = allocation.bytes;
        for (int id : allocation.tensor_ids) {
            if (temporary_ids.find(id) != temporary_ids.end()) arena_allocation.temporary_tensor_ids.push_back(id);
        }
        if (arena_allocation.temporary_tensor_ids.empty() || arena_allocation.bytes == 0) continue;
        total_bytes = align_up(total_bytes, kArenaAlignment);
        arena_allocation.offset = total_bytes;
        total_bytes += align_up(arena_allocation.bytes, kArenaAlignment);
        arena_allocations.push_back(std::move(arena_allocation));
    }
    if (arena_allocations.empty() || total_bytes == 0) return;

    cl_int err = CL_SUCCESS;
    cl_mem root = clCreateBuffer(state->context, CL_MEM_READ_WRITE, total_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !root) return;
    using MemHandle = std::remove_pointer_t<cl_mem>;
    auto root_owner = std::shared_ptr<MemHandle>(root, [](MemHandle* mem) {
        if (mem) clReleaseMemObject(mem);
    });
    arena_roots_.push_back(root_owner);

    for (const auto& allocation : arena_allocations) {
        cl_buffer_region region{allocation.offset, allocation.bytes};
        cl_mem view = clCreateSubBuffer(root,
                                        CL_MEM_READ_WRITE,
                                        CL_BUFFER_CREATE_TYPE_REGION,
                                        &region,
                                        &err);
        if (err != CL_SUCCESS || !view) {
            arena_mem_.clear();
            arena_views_.clear();
            arena_roots_.clear();
            arena_bytes_ = 0;
            return;
        }
        auto view_owner = std::shared_ptr<MemHandle>(view, [](MemHandle* mem) {
            if (mem) clReleaseMemObject(mem);
        });
        for (int id : allocation.temporary_tensor_ids) arena_mem_[id] = view;
        arena_views_.push_back(std::move(view_owner));
    }
    arena_bytes_ = total_bytes;
}

GraphExecutor compile_graph_executor(const CapturedGraph& graph, const GraphOptimizeOptions& options) {
    return GraphExecutor(graph, options);
}

} // namespace motifcl::autograd
