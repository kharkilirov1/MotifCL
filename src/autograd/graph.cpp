#include <motifcl/autograd/graph.hpp>

#include <motifcl/core/error.hpp>

#include <algorithm>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace motifcl::autograd {

namespace {
thread_local bool g_graph_capture_active = false;
thread_local CapturedGraph g_graph_capture;
thread_local std::vector<std::function<void()>> g_pending_replay_launches;

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
}

void GraphNodeInfo::replay() const {
    MCL_CHECK(replayable && replay_fn, "graph node is not replayable: " + op);
    replay_fn();
}

bool CapturedGraph::replayable() const {
    return std::all_of(nodes_.begin(), nodes_.end(), [](const GraphNodeInfo& node) {
        return node.replayable;
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
    MCL_CHECK(replayable(), "captured graph contains nodes that cannot be replayed");
    for (std::size_t index : schedule()) {
        nodes_.at(index).replay();
    }
}

void CapturedGraph::remember_tensor_specs(const std::vector<int>& ids) {
    std::lock_guard<std::mutex> lock(g_tensor_registry_mutex);
    for (int id : ids) {
        auto it = g_tensor_registry.find(id);
        if (it != g_tensor_registry.end()) tensor_specs_[id] = it->second;
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
        node.replay_fn = [launches = std::move(launches)]() {
            for (const auto& launch : launches) launch();
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
        node.replay_fn = [launches = std::move(launches), replay = std::move(replay)]() {
            for (const auto& launch : launches) launch();
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
    g_pending_replay_launches.push_back(std::move(replay));
}

void register_tensor(int id, const Shape& shape, DType dtype, std::size_t nbytes) {
    TensorSpec spec;
    spec.id = id;
    spec.shape = shape;
    spec.dtype = dtype;
    spec.nbytes = nbytes;
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

} // namespace motifcl::autograd
