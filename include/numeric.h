#ifndef _NUMERIC_H_
#define _NUMERIC_H_
#include "symbolic.h"
#include <memory>
#include <string>

using namespace std;

// Scan uses only the sparse structure. Mapped additionally stores the target
// position of every structural update, shared by all copies of the template.
enum class UpdateMode { Scan, Mapped };

// Single-device execution. Templates own immutable input/structure; tasks retain
// their template and own all mutable buffers. GPU errors throw runtime_error.
class GpuMatrix {
public:
    explicit GpuMatrix(const Symbolic_Matrix&, unsigned tmp_slots = 32, bool perturb = false,
                       UpdateMode update_mode = UpdateMode::Scan);
    ~GpuMatrix();
    unsigned tmp_slots() const; // Effective scratch slots; zero for Mapped.
    size_t task_device_bytes(unsigned batch_size = 1) const; // Exact buffers for one batch.
    size_t task_host_bytes() const;
    // Device allocation payloads, excluding allocator/stream/Graph overhead.
    // shared_device_bytes includes update_map_bytes; it is paid once per template.
    size_t shared_device_bytes() const;
    size_t update_map_bytes() const;
    double update_map_setup_ms() const; // Counting, construction and upload; zero for Scan.
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
    explicit GpuTask(const GpuMatrix&, unsigned batch_size = 1);
    unsigned batch_size() const;
    // Execution geometry diagnostics; identity layout reports stride zero.
    unsigned update_copy_tile_size() const;
    unsigned wide_copy_tile_size() const;
    unsigned soa_value_stride() const;
    bool normalizes_each_level() const;
    ~GpuTask();
    GpuTask(const GpuTask&) = delete;
    GpuTask& operator=(const GpuTask&) = delete;
    // Capture numeric work and output. Each enqueue restores host input outside
    // the graph, preserving stream ordering. Input is immutable for task lifetime.
    // Explicit preparation, outside measured work; no queued input or outstanding enqueue.
    void prepare_graph();
    // One outstanding enqueue per task. wait() completes H2D, compute and D2H.
    void enqueue();
    // Optionally queue input early. enqueue() otherwise uploads it automatically.
    // One call per completed iteration; invalidates previous factor results.
    void upload_input();
    void wait();
    const REAL* factors(unsigned copy = 0) const;
    void check() const; // finite factors and nonzero diagonal; after wait()
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Compatibility entry: one prepared template, one task, one execution.
bool LUonDevice(Symbolic_Matrix &, ostream &, ostream &, bool);

#endif
