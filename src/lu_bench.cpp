#include "numeric.h"
#include "preprocess.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}
struct Options {
    std::vector<std::string> inputs;
    size_t tasks = 0, warmup = 5, loops = 30;
    unsigned slots = 32;
    std::string output;
};
size_t number(const std::string& value, bool zero = false) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected an unsigned integer: " + value);
    size_t result=0;
    for (char c : value) {
        if (result > (std::numeric_limits<size_t>::max()-size_t(c-'0'))/10)
            throw std::invalid_argument("integer overflow: " + value);
        result = result*10 + size_t(c-'0');
    }
    if (!zero && !result) throw std::invalid_argument("value must be positive");
    return result;
}
Options parse(int argc, char** argv) {
    Options o;
    for (int i=1; i<argc; ++i) {
        const std::string key=argv[i];
        if (i+1 == argc) throw std::invalid_argument("missing value for " + key);
        const std::string value=argv[++i];
        if (key == "-i") o.inputs.push_back(value);
        else if (key == "--tasks") o.tasks=number(value);
        else if (key == "--warmup") o.warmup=number(value,true);
        else if (key == "--loops") o.loops=number(value);
        else if (key == "--tmp-slots") {
            const size_t slots=number(value);
            if (slots > std::numeric_limits<unsigned>::max()) throw std::invalid_argument("tmp slots overflow");
            o.slots=unsigned(slots);
        }
        else if (key == "--output") o.output=value;
        else throw std::invalid_argument("unknown option: " + key);
    }
    if (o.inputs.empty()) throw std::invalid_argument("at least one -i input is required");
    if (!o.tasks) o.tasks=o.inputs.size();
    if (o.tasks < o.inputs.size()) throw std::invalid_argument("tasks must cover every input template");
    return o;
}
struct NicsState {
    SNicsLU state{};
    bool initialized=false;
    ~NicsState() { if (initialized) NicsLU_Destroy(&state); }
};
struct InputArrays {
    double* values=nullptr;
    unsigned *rows=nullptr, *columns=nullptr;
    ~InputArrays() { free(values); free(rows); free(columns); }
};
struct Matrix {
    NicsState nics;
    std::unique_ptr<Symbolic_Matrix> symbolic;
    std::unique_ptr<GpuMatrix> gpu;
    Matrix(const std::string& path, unsigned slots) {
        InputArrays a;
        if (preprocess(path.c_str(), &nics.state, &a.values, &a.rows, &a.columns))
            throw std::runtime_error("preprocess failed: " + path);
        nics.initialized=true;
        symbolic.reset(new Symbolic_Matrix(nics.state.n,std::cout,std::cerr));
        symbolic->fill_in(a.rows,a.columns);
        symbolic->csr();
        symbolic->predictLU(a.rows,a.columns,a.values);
        symbolic->leveling();
        gpu.reset(new GpuMatrix(*symbolic,slots));
    }
};


std::string quoted(const std::string& s) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : s) {
        if (c=='"' || c=='\\') out << '\\' << char(c);
        else if (c<32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else out << char(c);
    }
    out << '"';
    return out.str();
}
void add_bytes(size_t& total, size_t value) {
    if (value > std::numeric_limits<size_t>::max()-total) throw std::overflow_error("task memory size overflow");
    total += value;
}
size_t multiplied_bytes(size_t count, size_t bytes) {
    if (bytes && count > std::numeric_limits<size_t>::max()/bytes)
        throw std::overflow_error("task memory size overflow");
    return count*bytes;
}
}

int main(int argc, char** argv) {
    if (argc==2 && std::string(argv[1])=="--help") {
        std::cout << "Usage: glu_bench -i MATRIX [-i MATRIX ...] [--tasks N]\n"
                  << "  [--warmup 5] [--loops 30] [--tmp-slots 32] [--output result.json]\n"
                  << "One stream per task; round-robin templates; no perturbation.\n";
        return 0;
    }
    try {
        const Options o=parse(argc,argv);
        const auto setup_start=Clock::now();
        std::vector<std::unique_ptr<Matrix>> matrices;
        for (const auto& path : o.inputs) matrices.emplace_back(new Matrix(path,o.slots));
        size_t device_bytes=0, pinned_bytes=0;
        for (const auto& m : matrices) add_bytes(pinned_bytes,m->gpu->task_host_bytes());
        for (size_t i=0; i<matrices.size(); ++i) {
            const auto& g=*matrices[i]->gpu;
            const size_t count=o.tasks/matrices.size()+(i<o.tasks%matrices.size());
            add_bytes(device_bytes,multiplied_bytes(count,g.task_device_bytes()));
            add_bytes(pinned_bytes,multiplied_bytes(count,g.task_host_bytes()));
        }
        const size_t available=GpuMatrix::free_device_bytes();
        std::cout << "Task device buffers: " << device_bytes << " bytes; pinned buffers: " << pinned_bytes << " bytes\n";
        if (device_bytes > available)
            throw std::runtime_error("task buffers exceed free device memory; explicitly reduce tasks or tmp slots");
        std::vector<std::unique_ptr<GpuTask>> tasks;
        tasks.reserve(o.tasks);
        for (size_t i=0; i<o.tasks; ++i) tasks.emplace_back(new GpuTask(*matrices[i%matrices.size()]->gpu));
        const double setup_ms=elapsed(setup_start);
        auto loop = [&]() {
            for (auto& t : tasks) t->enqueue();
            for (auto& t : tasks) t->wait();
        };
        for (size_t i=0; i<o.warmup; ++i) loop();
        std::vector<double> loop_ms(o.loops);
        for (size_t i=0; i<o.loops; ++i) {
            const auto start=Clock::now();
            loop();
            loop_ms[i]=elapsed(start);
        }
        const auto check_start=Clock::now();
        const double tolerance=sizeof(REAL)==8 ? 1e-10 : 1e-3;
        double max_residual=0;
        for (size_t i=0; i<tasks.size(); ++i) {
            try {
                tasks[i]->check();
                auto& m=*matrices[i%matrices.size()];
                const REAL* factors=tasks[i]->factors();
                const double residual=m.symbolic->factor_residual(factors);
                if (!std::isfinite(residual) || residual>tolerance)
                    throw std::runtime_error("factor residual exceeds tolerance: " + std::to_string(residual));
                max_residual=std::max(max_residual,residual);
                const auto x=m.symbolic->solve(&m.nics.state,std::vector<REAL>(m.symbolic->n,REAL(1)),factors);
                for (REAL value : x) if (!std::isfinite(value)) throw std::runtime_error("nonfinite solution");
            } catch (const std::exception& error) {
                throw std::runtime_error("task " + std::to_string(i) + ": " + error.what());
            }
        }
        const double check_ms=elapsed(check_start);
        double total_ms=0;
        for (double ms : loop_ms) total_ms+=ms;
        std::ostringstream report;
        report << std::setprecision(17) << "{\n  \"schema\": 1, \"status\": \"ok\",\n  \"backend\": "
#if defined(GLU_USE_HIP)
               << "\"HIP\""
#else
               << "\"CUDA\""
#endif
               << ", \"real_bits\": " << sizeof(REAL)*8
               << ", \"device\": " << quoted(matrices[0]->gpu->device_name())
               << ",\n  \"schedule\": \"one-stream-per-task\", \"submission\": \"single-host-thread-task-order\","
               << "\n  \"tasks\": " << o.tasks << ", \"streams\": " << o.tasks
               << ", \"warmup\": " << o.warmup << ", \"loops\": " << o.loops
               << ", \"requested_tmp_slots\": " << o.slots
               << ",\n  \"task_device_bytes\": " << device_bytes << ", \"pinned_bytes\": " << pinned_bytes
               << ",\n  \"setup_ms\": " << setup_ms << ", \"validation_ms\": " << check_ms
               << ", \"ms_per_loop\": " << total_ms/o.loops
               << ",\n  \"validation\": \"all-tasks-final-loop-full-factor-residual-and-finite-solve\","
               << "\n  \"checked_tasks\": " << tasks.size() << ", \"max_factor_residual\": " << max_residual
               << ", \"factor_residual_tolerance\": " << tolerance
               << ",\n  \"templates\": [";
        for (size_t i=0; i<matrices.size(); ++i) {
            if (i) report << ',';
            const auto& m=*matrices[i];
            report << "{\"path\":" << quoted(o.inputs[i]) << ",\"n\":" << m.symbolic->n
                   << ",\"factor_nnz\":" << m.symbolic->nnz << ",\"tasks\":" << o.tasks/matrices.size()+(i<o.tasks%matrices.size())
                   << ",\"effective_tmp_slots\":" << m.gpu->tmp_slots() << '}';
        }
        report << "],\n  \"loop_ms\": [";
        for (size_t i=0; i<loop_ms.size(); ++i) { if (i) report << ','; report << loop_ms[i]; }
        report << "]\n}\n";
        if (!o.output.empty()) {
            std::ofstream file(o.output);
            file << report.str();
            file.close();
            if (!file) throw std::runtime_error("failed to write result: " + o.output);
        }
        std::cout << report.str();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
