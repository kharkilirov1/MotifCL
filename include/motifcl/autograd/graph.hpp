#pragma once

#include <cstddef>
#include <CL/cl.h>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <motifcl/core/dtype.hpp>
#include <motifcl/core/shape.hpp>

namespace motifcl {
class Tensor;
struct OpenCLContextState;
class DriverCommandBuffer;
}

namespace motifcl::autograd {

struct TensorSpec {
    int id = 0;
    Shape shape;
    DType dtype = DType::F32;
    std::size_t nbytes = 0;
    cl_mem mem = nullptr;
    std::shared_ptr<std::remove_pointer_t<cl_mem>> retained_mem;
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

struct GraphRuntimeBinding {
    int tensor_id = 0;
    std::size_t allocation_id = 0;
    std::size_t offset = 0;
    std::size_t nbytes = 0;
    Shape shape;
    DType dtype = DType::F32;
};

struct GraphRuntimePlan {
    GraphBufferPlan buffer_plan;
    std::vector<TensorSpec> runtime_specs;
    std::vector<GraphRuntimeBinding> bindings;
    bool compatible = false;
    bool kernel_rebinding = false;
    std::string note;
};

struct GraphOptimizeOptions {
    bool enable_buffer_reuse = true;
    bool shape_polymorphic = false;
};

struct GraphKernelLaunchInfo {
    std::string kernel_name;
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_kernel kernel = nullptr;
    cl_uint work_dim = 0;
    std::vector<std::size_t> global_work_size;
    std::vector<std::size_t> local_work_size;
    bool has_local_work_size = false;
    std::vector<std::pair<int, cl_mem>> buffer_args;
    std::vector<std::pair<int, int>> tensor_arg_bindings;
    std::shared_ptr<::motifcl::OpenCLContextState> retained_state;
    std::shared_ptr<void> retained_kernel;
};

struct GraphNodeInfo {
    std::string op;
    std::vector<int> inputs;
    std::vector<int> outputs;
    std::vector<int> temporaries;
    bool replayable = false;
    bool rebindable = false;
    bool command_buffer_recordable = false;
    const std::vector<GraphKernelLaunchInfo>& kernel_launches() const { return kernel_launches_; }

    void replay() const;
    void replay(const std::unordered_map<int, cl_mem>& tensor_bindings) const;

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
    std::function<void(const std::unordered_map<int, cl_mem>&)> replay_fn;
    std::vector<GraphKernelLaunchInfo> kernel_launches_;
};

class CapturedGraph {
public:
    const std::vector<GraphNodeInfo>& nodes() const { return nodes_; }
    std::vector<GraphNodeInfo>& nodes() { return nodes_; }
    std::size_t size() const { return nodes_.size(); }
    bool empty() const { return nodes_.empty(); }
    bool replayable() const;
    bool rebindable() const;
    std::vector<std::size_t> schedule() const;
    const std::unordered_map<int, TensorSpec>& tensor_specs() const { return tensor_specs_; }
    std::vector<TensorSpec> tensor_specs_list() const;
    GraphBufferPlan plan_buffers(const GraphOptimizeOptions& options = GraphOptimizeOptions{}) const;
    GraphBufferPlan optimize(const GraphOptimizeOptions& options = GraphOptimizeOptions{}) const;
    GraphRuntimePlan compile_runtime_plan(const std::vector<TensorSpec>& runtime_specs,
                                          const GraphOptimizeOptions& options = GraphOptimizeOptions{}) const;
    CapturedGraph shape_polymorphic() const;
    bool compatible_with(const std::vector<TensorSpec>& runtime_specs, bool allow_dynamic_dims = true) const;
    void replay() const;
    void replay(const std::unordered_map<int, cl_mem>& tensor_bindings) const;
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
void record_kernel_launch(const std::string& kernel_name,
                          std::vector<std::pair<int, cl_mem>> buffer_args,
                          std::function<void(const std::unordered_map<int, cl_mem>&,
                                             const std::vector<std::pair<int, int>>&)>
                              replay);
void record_kernel_launch(GraphKernelLaunchInfo launch,
                          std::function<void(const std::unordered_map<int, cl_mem>&,
                                             const std::vector<std::pair<int, int>>&)>
                              replay);
void register_tensor(int id, const Shape& shape, DType dtype, std::size_t nbytes, cl_mem mem = nullptr);

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

class GraphExecutor {
public:
    explicit GraphExecutor(CapturedGraph graph,
                           const GraphOptimizeOptions& options = GraphOptimizeOptions{});
    ~GraphExecutor();
    GraphExecutor(const GraphExecutor&) = delete;
    GraphExecutor& operator=(const GraphExecutor&) = delete;
    GraphExecutor(GraphExecutor&&) noexcept;
    GraphExecutor& operator=(GraphExecutor&&) noexcept;

    const CapturedGraph& graph() const { return graph_; }
    const GraphRuntimePlan& runtime_plan() const { return runtime_plan_; }
    bool replayable() const { return graph_.replayable(); }
    std::size_t executions() const { return executions_; }
    bool rebindable() const { return graph_.rebindable(); }
    const std::string& execution_mode() const { return execution_mode_; }
    void bind_tensor(int captured_tensor_id, const Tensor& tensor);
    void clear_bindings();
    std::size_t bound_tensor_count() const { return bound_tensors_.size(); }
    std::size_t arena_binding_count() const { return arena_mem_.size(); }
    std::size_t arena_bytes() const { return arena_bytes_; }
    void execute();

private:
    CapturedGraph graph_;
    GraphOptimizeOptions options_;
    GraphRuntimePlan runtime_plan_;
    std::vector<std::size_t> cached_schedule_;
    std::unordered_map<int, cl_mem> bound_mem_;
    std::unordered_map<int, cl_mem> arena_mem_;
    std::unordered_map<int, std::shared_ptr<Tensor>> bound_tensors_;
    std::vector<std::shared_ptr<std::remove_pointer_t<cl_mem>>> arena_roots_;
    std::vector<std::shared_ptr<std::remove_pointer_t<cl_mem>>> arena_views_;
    std::unique_ptr<::motifcl::DriverCommandBuffer> driver_command_buffer_;
    std::string execution_mode_ = "host_replay_fallback";
    std::size_t arena_bytes_ = 0;
    std::size_t executions_ = 0;

    void initialize_temporary_arena();
};

GraphExecutor compile_graph_executor(const CapturedGraph& graph,
                                     const GraphOptimizeOptions& options = GraphOptimizeOptions{});

} // namespace motifcl::autograd
