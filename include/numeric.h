#ifndef _NUMERIC_H_
#define _NUMERIC_H_
#include "symbolic.h"
#include <memory>
#include <string>

using namespace std;

// Execution strategy; its representation is shared by independent value copies.
enum class UpdateMode
{
    Scan,
    Mapped,
    Panel,
    PanelTiled,
    Fused // One matrix per block, executing the non-dense Panel stages together.
};

// A snapshot is the immutable default input, restored before each factorization.
// Caller-provided values bypass this policy and never replace the snapshot.
enum class SnapshotPolicy
{
    PerCopy, // Upload the immutable host snapshot separately for every copy.
    Shared,  // Upload once per batch, then broadcast; not GPU shared memory.
    Device,  // Retain one read-only device initial vector per template.
    Auto     // Shared for multi-copy Panel; PerCopy otherwise.
};

// Single-device execution. Templates own immutable input/structure; tasks retain
// their template and own all mutable buffers. GPU errors throw runtime_error.
// Construction copies the required structure, default values and compact order;
// the caller need not retain the source Symbolic_Matrix or input-position vector.
class GpuMatrix
{
public:
    explicit GpuMatrix(const Symbolic_Matrix&,
        unsigned tmp_slots = 32,
        bool perturb = false,
        UpdateMode update_mode = UpdateMode::Scan,
        SnapshotPolicy snapshot_policy = SnapshotPolicy::Auto);
    // Fixed compact order: nonempty, unique indices < factor_nnz, including
    // explicit zeros that may change. Omitted slots are zero on each compact upload.
    // Caller performs any preprocessing/permutation/scaling before supplying values.
    GpuMatrix(const Symbolic_Matrix&,
        const std::vector<unsigned>& input_positions,
        unsigned tmp_slots = 32,
        bool perturb = false,
        UpdateMode update_mode = UpdateMode::Scan,
        SnapshotPolicy snapshot_policy = SnapshotPolicy::Auto);
    ~GpuMatrix();
    unsigned tmp_slots() const;                              // Effective scratch slots; zero outside Scan.
    size_t task_device_bytes(unsigned batch_size = 1) const; // Exact buffers for one batch.
    size_t task_host_bytes() const;                          // Bytes of one pinned input/output vector.
    SnapshotPolicy snapshot_policy() const; // Requested policy; Auto resolves separately for each task.
    double snapshot_setup_ms() const;       // Device initial allocation and upload only.
    size_t compact_input_elements() const;  // Per copy; zero without a compact order.
    double compact_input_setup_ms() const;  // Validation and position-map upload.
    // Device allocation payloads, excluding allocator/stream/Graph overhead.
    // Shared bytes include maps/plans, compact positions and a Device snapshot,
    // when enabled, paid once per template.
    size_t shared_device_bytes() const;
    size_t update_map_bytes() const;
    double update_map_setup_ms() const; // Counting, construction and upload; zero outside Mapped.
    size_t update_plan_bytes() const;   // Prepared update-plan device payload, including Fused metadata.
    double update_plan_setup_ms() const;
    bool uses_native_mmac() const;
    std::string device_name() const;
    static size_t free_device_bytes();
    static void select_device(); // Select visible GPU 0 for the calling host thread.
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class GpuTask;
    friend class GpuTaskSet;
};

class GpuTask
{
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
    SnapshotPolicy snapshot_policy() const; // Effective policy, fixed at construction; never Auto.
    // Execution geometry diagnostics; identity layout reports stride zero.
    unsigned update_copy_tile_size() const;
    unsigned wide_copy_tile_size() const;
    unsigned soa_value_stride() const;
    bool normalizes_each_level() const;
    size_t numeric_launches() const;    // Numeric work and output conversion; excludes input preparation.
    size_t numeric_wide_blocks() const; // Scan/Mapped only; zero means not applicable for panel modes.
    // Default snapshot diagnostics; explicit fresh/compact uploads bypass these.
    size_t snapshot_h2d_bytes() const;
    size_t snapshot_h2d_ops() const;
    size_t snapshot_restore_launches() const; // Default-input kernels, including any input layout conversion.
    ~GpuTask();
    GpuTask(const GpuTask&) = delete;
    GpuTask& operator=(const GpuTask&) = delete;

    // Capture numeric work and output, preserving stable device addresses.
    // All input paths prepare the execution layout outside the graph.
    // Explicit preparation, outside measured work; no queued input or outstanding enqueue.
    // Repeated calls in this state reuse the prepared graph without recapturing.
    void prepare_graph();

    // One outstanding enqueue per task. wait() completes H2D, compute and D2H.
    // Each enqueue uploads the default input unless explicit input is queued.
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

    // AoS [copy][fixed compact input position]. Exactly compact_input_elements
    // * copies values. Same source lifetime, state, and perturbation constraints
    // as fresh upload_input. Omitted slots reset to zero; the default is unchanged.
    void upload_compact_input(const REAL* values, size_t elements);

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

// One ordered collection of independent, possibly different Fused templates.
// Owns one stream and contiguous AoS input/output buffers; shares template plans.
// Calls on a set are serialized, with the same source/result lifetime and failure
// rules as GpuTask. Template wrappers may be destroyed after construction.
class GpuTaskSet
{
public:
    // task_templates[t] indexes templates. Repeated indices create independent
    // numeric tasks sharing a template. The lists are only needed during construction.
    GpuTaskSet(const std::vector<const GpuMatrix*>& templates, const std::vector<size_t>& task_templates);
    ~GpuTaskSet();
    GpuTaskSet(const GpuTaskSet&) = delete;
    GpuTaskSet& operator=(const GpuTaskSet&) = delete;
    size_t task_count() const;
    // Packed in logical task order. Offset(task_count()) is the total size.
    size_t factor_offset(size_t task) const;
    size_t compact_input_offset(size_t task) const;
    size_t device_bytes() const; // Set-owned arrays; shared template plans excluded.
    size_t pinned_bytes() const; // Output plus the default cache, once allocated.
    double default_input_setup_ms() const;
    // Default restoration lazily packs immutable host snapshots once per set.
    // Explicit inputs contain exactly offset(task_count()) elements, bypass
    // that cache and reject templates with static pivot perturbation.
    void upload_input();
    void upload_input(const REAL* values, size_t elements);
    void upload_compact_input(const REAL* values, size_t elements);
    void enqueue(); // Default input is restored when none is queued.
    void wait();    // All factors are on the host before returning.
    const REAL* factors(size_t task) const;
    void check() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Single-matrix convenience entry used by the solver CLI.
bool LUonDevice(Symbolic_Matrix&, ostream&, ostream&, bool);

#endif
