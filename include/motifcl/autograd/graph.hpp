#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <motifcl/core/dtype.hpp>
#include <motifcl/core/shape.hpp>

namespace motifcl::autograd {

struct TensorSpec {
    int id = 0;
    Shape shape;
    DType dtype = DType::F32;
    std::size_t nbytes = 0;
    bool dynamic_shape = false;
};

struct GraphBufferAllocation {
    std::size_t allocation_id = 0;
    std::vector<int> tensor_ids;
    std::size_t bytes = 0;
    std::size_t first_node = 0;
    std::size_t last_node = 0;
};

struct GraphBufferPlan {
    std::vector<GraphBufferAllocation> allocations;
    std::size_t peak_bytes = 0;
    std::size_t total_output_bytes = 0;
    std::size_t reused_bytes = 0;
    bool shape_polymorphic = false;
};

struct GraphOptimizeOptions {
    bool enable_buffer_reuse = true;
    bool shape_polymorphic = false;
};

struct GraphNodeInfo {
    std::string op;
    std::vector<int> inputs;
    std::vector<int> outputs;
    std::vector<int> temporaries;
    bool replayable = false;

    void replay() const;

private:
    friend class CapturedGraph;
    friend void record_op(const std::string& op, std::vector<int> inputs, std::vector<int> outputs);
    friend void record_op(const std::string& op, std::vector<int> inputs, std::vector<int> outputs, bool replayable);
    friend void record_replay_op(const std::string& op,
                                 std::vector<int> inputs,
                                 std::vector<int> outputs,
                                 std::function<void()> replay);
    friend void record_replay_op(const std::string& op,
                                 std::vector<int> inputs,
                                 std::vector<int> outputs,
                                 std::vector<int> temporaries,
                                 std::function<void()> replay);
    std::function<void()> replay_fn;
};

class CapturedGraph {
public:
    const std::vector<GraphNodeInfo>& nodes() const { return nodes_; }
    std::vector<GraphNodeInfo>& nodes() { return nodes_; }
    std::size_t size() const { return nodes_.size(); }
    bool empty() const { return nodes_.empty(); }
    bool replayable() const;
    std::vector<std::size_t> schedule() const;
    const std::unordered_map<int, TensorSpec>& tensor_specs() const { return tensor_specs_; }
    std::vector<TensorSpec> tensor_specs_list() const;
    GraphBufferPlan plan_buffers(const GraphOptimizeOptions& options = GraphOptimizeOptions{}) const;
    GraphBufferPlan optimize(const GraphOptimizeOptions& options = GraphOptimizeOptions{}) const;
    CapturedGraph shape_polymorphic() const;
    bool compatible_with(const std::vector<TensorSpec>& runtime_specs, bool allow_dynamic_dims = true) const;
    void replay() const;
    void execute() const { replay(); }
    void clear() {
        nodes_.clear();
        tensor_specs_.clear();
    }

private:
    friend void record_op(const std::string& op, std::vector<int> inputs, std::vector<int> outputs, bool replayable);
    friend void record_replay_op(const std::string& op,
                                 std::vector<int> inputs,
                                 std::vector<int> outputs,
                                 std::vector<int> temporaries,
                                 std::function<void()> replay);

    std::vector<GraphNodeInfo> nodes_;
    std::unordered_map<int, TensorSpec> tensor_specs_;

    void remember_tensor_specs(const std::vector<int>& ids);
};

void begin_graph_capture();
CapturedGraph end_graph_capture();
bool is_graph_capturing();
const CapturedGraph& current_graph_capture();
void clear_graph_capture();
void record_op(const std::string& op, std::vector<int> inputs, std::vector<int> outputs);
void record_op(const std::string& op, std::vector<int> inputs, std::vector<int> outputs, bool replayable);
void record_replay_op(const std::string& op,
                      std::vector<int> inputs,
                      std::vector<int> outputs,
                      std::function<void()> replay);
void record_replay_op(const std::string& op,
                      std::vector<int> inputs,
                      std::vector<int> outputs,
                      std::vector<int> temporaries,
                      std::function<void()> replay);
void record_kernel_launch(const std::string& kernel_name, std::function<void()> replay);
void register_tensor(int id, const Shape& shape, DType dtype, std::size_t nbytes);

class GraphCaptureGuard {
public:
    GraphCaptureGuard();
    GraphCaptureGuard(const GraphCaptureGuard&) = delete;
    GraphCaptureGuard& operator=(const GraphCaptureGuard&) = delete;
    ~GraphCaptureGuard();
    CapturedGraph finish();

private:
    bool active_ = false;
};

} // namespace motifcl::autograd
