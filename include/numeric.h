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
    explicit GpuTask(const GpuMatrix&);
    ~GpuTask();
    GpuTask(const GpuTask&) = delete;
    GpuTask& operator=(const GpuTask&) = delete;
    // One outstanding enqueue per task. wait() completes H2D, compute and D2H.
    void enqueue();
    void wait();
    const REAL* factors() const;
    void check() const; // finite factors and nonzero diagonal; after wait()
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Compatibility entry: one prepared template, one task, one execution.
bool LUonDevice(Symbolic_Matrix &, ostream &, ostream &, bool);

#endif
