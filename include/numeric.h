#ifndef _NUMERIC_H_
#define _NUMERIC_H_
#include "symbolic.h"
#include <memory>
#include <string>

using namespace std;

// Each strategy owns only its execution representation, shared by value copies.
// PanelTiled extracts dense rectangles; supported targets use native MMAC,
// other GPU targets use scalar updates.
enum class UpdateMode { Scan, Mapped, Panel, PanelTiled };

// Single-device execution. Templates own immutable input/structure; tasks retain
// their template and own all mutable buffers. GPU errors throw runtime_error.
class GpuMatrix {
public:
    explicit GpuMatrix(const Symbolic_Matrix&, unsigned tmp_slots = 32, bool perturb = false,
                       UpdateMode update_mode = UpdateMode::Scan);
    ~GpuMatrix();
    unsigned tmp_slots() const; // Effective scratch slots; zero outside Scan.
    size_t task_device_bytes(unsigned batch_size = 1) const; // Exact buffers for one batch.
    size_t task_host_bytes() const; // Bytes of one pinned input/output vector.
    // Device allocation payloads, excluding allocator/stream/Graph overhead.
    // Shared bytes include map or panel-plan bytes, paid once per template.
    size_t shared_device_bytes() const;
    size_t update_map_bytes() const;
    double update_map_setup_ms() const; // Counting, construction and upload; zero outside Mapped.
    size_t update_plan_bytes() const; // Panel groups/pairs and optional dense indices.
    double update_plan_setup_ms() const;
    bool uses_native_mmac() const;
    std::string device_name() const;
    static size_t free_device_bytes();
    static void select_device(); // Select visible GPU 0 for the calling host thread.
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class GpuTask;
};

class GpuTask {
public:
    // Independent copies of one immutable template; each copy owns its values,
    // output and (only in scan mode) scratch. One stream runs the entire batch.
    // Tasks keep the template alive after its GpuMatrix wrapper is destroyed.
    // Serialize calls on the same task; different tasks may use different host
    // threads. Each submitting thread must first call GpuMatrix::select_device().
    // Rebuild the task after a GPU submission or execution failure.
    // Destruction drains even partially submitted work before releasing buffers.
    explicit GpuTask(const GpuMatrix&, unsigned batch_size = 1);
    unsigned batch_size() const;
    // Execution geometry diagnostics; identity layout reports stride zero.
    unsigned update_copy_tile_size() const;
    unsigned wide_copy_tile_size() const;
    unsigned soa_value_stride() const;
    bool normalizes_each_level() const;
    size_t numeric_launches() const;
    size_t numeric_wide_blocks() const; // Scan/Mapped only; zero means not applicable for panel modes.
    ~GpuTask();
    GpuTask(const GpuTask&) = delete;
    GpuTask& operator=(const GpuTask&) = delete;
    // Capture numeric work and output, preserving stable device addresses.
    // Each enqueue uploads the default input unless explicit input is queued.
    // Explicit preparation, outside measured work; no queued input or outstanding enqueue.
    // Repeated calls in this state reuse the prepared graph without recapturing.
    void prepare_graph();
    // One outstanding enqueue per task. wait() completes H2D, compute and D2H.
    void enqueue();
    // Optionally queue input early. enqueue() otherwise uploads it automatically.
    // Queue at most once before enqueue; this invalidates previous factor results.
    // An upload alone is not a result: enqueue, then wait before reading factors.
    void upload_input();
    // Explicit values in fixed symbolic factor-slot order, AoS across copies.
    // Exactly factor_nnz * batch_size elements. Keep host storage readable and
    // unchanged through enqueue()+wait(); pinned storage permits async upload.
    // Reuses the prepared Graph and does not replace the template's default.
    // Not available with perturbation (whose threshold belongs to the snapshot).
    // On submission failure, attempt to drain before rethrowing; rebuild the task.
    void upload_input(const REAL* values, size_t elements);
    // Complete submitted work and propagate asynchronous errors. Upload-only
    // state is rejected because numeric work and output have not been enqueued.
    void wait();
    // Task-owned view, readable after a successful wait. Valid until the next
    // upload/enqueue or task destruction; copy selects one independent factor.
    const REAL* factors(unsigned copy = 0) const;
    void check() const; // finite factors and nonzero diagonal; after wait()
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Compatibility entry: one prepared template, one task, one execution.
bool LUonDevice(Symbolic_Matrix &, ostream &, ostream &, bool);

#endif
