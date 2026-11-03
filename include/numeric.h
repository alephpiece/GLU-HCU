#ifndef _NUMERIC_H_
#define _NUMERIC_H_
#include "symbolic.h"
#include <memory>
#include <string>

using namespace std;

// Single-device execution. Templates own immutable input/structure; tasks retain
// their template and own all mutable buffers. GPU errors throw runtime_error.
class GpuMatrix {
public:
    explicit GpuMatrix(const Symbolic_Matrix&, unsigned tmp_slots = 32, bool perturb = false);
    ~GpuMatrix();
    unsigned tmp_slots() const;
    size_t task_device_bytes() const;
    size_t task_host_bytes() const;
    std::string device_name() const;
    static size_t free_device_bytes();
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class GpuTask;
};

class GpuTask {
public:
    // Independent copies of one immutable template; each copy owns its values,
    // scratch and output. One stream runs the entire batch.
    explicit GpuTask(const GpuMatrix&, unsigned batch_size = 1);
    unsigned batch_size() const;
    ~GpuTask();
    GpuTask(const GpuTask&) = delete;
    GpuTask& operator=(const GpuTask&) = delete;
    // Capture the same H2D/compute/D2H chain for subsequent enqueues.
    // Explicit preparation, outside measured work; no outstanding enqueue.
    void prepare_graph();
    // One outstanding enqueue per task. wait() completes H2D, compute and D2H.
    void enqueue();
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
