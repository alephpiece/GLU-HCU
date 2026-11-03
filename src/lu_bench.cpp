#include "numeric.h"
#include "preprocess.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <map>
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
    std::vector<size_t> task_templates, counts;
    std::string workload;
    size_t tasks = 0, warmup = 5, loops = 30;
    unsigned slots = 32, batch_size = 1;
    std::string output;
    std::string check = "full";
    bool diagnostics = false;
    bool graph = false;
    size_t template_index(size_t task) const {
        return task_templates.empty() ? task%inputs.size() : task_templates[task];
    }
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
// The workload describes independent logical tasks, not a scheduling algorithm.
void read_workload(Options& o) {
    std::ifstream file(o.workload);
    if (!file) throw std::invalid_argument("cannot open workload: " + o.workload);
    const auto directory=std::filesystem::path(o.workload).parent_path();
    std::map<std::string,size_t> ids;
    std::string line;
    size_t line_number=0;
    bool header=false, tasks_started=false;
    while (std::getline(file,line)) {
        ++line_number;
        std::istringstream in(line);
        std::string kind, id, path, extra;
        if (!(in >> kind) || kind[0]=='#') continue;
        try {
            if (!header) {
                if (kind!="glu-workload" || !(in >> id) || id!="1" || (in >> extra))
                    throw std::invalid_argument("expected glu-workload 1");
                header=true;
            } else if (kind=="matrix") {
                if (tasks_started || !(in >> id >> std::quoted(path)) || path.empty() || (in >> extra))
                    throw std::invalid_argument("expected matrix ID PATH before all tasks");
                if (!ids.emplace(id,o.inputs.size()).second)
                    throw std::invalid_argument("duplicate matrix ID: " + id);
                o.inputs.push_back((directory/std::filesystem::path(path)).lexically_normal().string());
            } else if (kind=="task") {
                if (!(in >> id) || (in >> extra) || !ids.count(id))
                    throw std::invalid_argument("expected task with a defined matrix ID");
                tasks_started=true;
                o.task_templates.push_back(ids.at(id));
            } else throw std::invalid_argument("unknown workload record: " + kind);
        } catch (const std::exception& error) {
            throw std::invalid_argument(o.workload + ":" + std::to_string(line_number) + ": " + error.what());
        }
    }
    if (file.bad()) throw std::runtime_error("failed reading workload: " + o.workload);
    if (!header || o.task_templates.empty()) throw std::invalid_argument("empty workload");
    o.tasks=o.task_templates.size();
}
Options parse(int argc, char** argv) {
    Options o;
    for (int i=1; i<argc; ++i) {
        const std::string key=argv[i];
        if (key == "--diagnostics") { o.diagnostics=true; continue; }
        if (key == "--graph") { o.graph=true; continue; }
        if (i+1 == argc) throw std::invalid_argument("missing value for " + key);
        const std::string value=argv[++i];
        if (key == "-i") o.inputs.push_back(value);
        else if (key == "--workload") {
            if (value.empty()) throw std::invalid_argument("empty workload path");
            o.workload=value;
        }
        else if (key == "--tasks") o.tasks=number(value);
        else if (key == "--warmup") o.warmup=number(value,true);
        else if (key == "--loops") o.loops=number(value);
        else if (key == "--batch-size") {
            const size_t count=number(value);
            if (count>std::numeric_limits<unsigned>::max()) throw std::invalid_argument("batch size overflow");
            o.batch_size=unsigned(count);
        }
        else if (key == "--tmp-slots") {
            const size_t slots=number(value);
            if (slots > std::numeric_limits<unsigned>::max()) throw std::invalid_argument("tmp slots overflow");
            o.slots=unsigned(slots);
        }
        else if (key == "--output") o.output=value;
        else if (key == "--check") {
            if (value != "full" && value != "basic") throw std::invalid_argument("check must be full or basic");
            o.check=value;
        }
        else throw std::invalid_argument("unknown option: " + key);
    }
    if (!o.workload.empty()) {
        if (!o.inputs.empty() || o.tasks) throw std::invalid_argument("--workload cannot be combined with -i or --tasks");
        read_workload(o);
    } else {
        if (o.inputs.empty()) throw std::invalid_argument("at least one -i input is required");
        if (!o.tasks) o.tasks=o.inputs.size();
        if (o.tasks < o.inputs.size()) throw std::invalid_argument("tasks must cover every input template");
    }
    o.counts.assign(o.inputs.size(),0);
    if (o.workload.empty()) {
        for (size_t i=0; i<o.inputs.size(); ++i) o.counts[i]=o.tasks/o.inputs.size()+(i<o.tasks%o.inputs.size());
    } else for (size_t index : o.task_templates) ++o.counts[index];
    for (size_t count : o.counts) if (!count) throw std::invalid_argument("unused workload matrix");
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
        std::cout << "Usage: glu_bench (-i MATRIX [-i MATRIX ...] [--tasks N] | --workload FILE)\n"
                  << "  [--warmup 5] [--loops 30] [--tmp-slots 32] [--output result.json]\n"
                  << "  [--check full|basic] [--diagnostics] [--graph] [--batch-size 1]\n"
                  << "One stream per homogeneous batch; explicit logical workload or round-robin inputs; no perturbation.\n";
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
            const size_t count=o.counts[i];
            add_bytes(device_bytes,multiplied_bytes(count,g.task_device_bytes()));
            add_bytes(pinned_bytes,multiplied_bytes(count,g.task_host_bytes()));
        }
        const size_t available=GpuMatrix::free_device_bytes();
        std::cout << "Task device buffers: " << device_bytes << " bytes; pinned buffers: " << pinned_bytes << " bytes\n";
        if (device_bytes > available)
            throw std::runtime_error("task buffers exceed free device memory; explicitly reduce tasks or tmp slots");
        std::vector<std::unique_ptr<GpuTask>> tasks;
        tasks.reserve(o.tasks);
        std::vector<size_t> task_template, template_batches(matrices.size(),0);
        auto add_batch = [&](size_t index, unsigned copies) {
            tasks.emplace_back(new GpuTask(*matrices[index]->gpu,copies));
            task_template.push_back(index);
            ++template_batches[index];
        };
        if (o.batch_size==1) {
            // Preserve the reference task order when batching is disabled.
            for (size_t i=0; i<o.tasks; ++i) add_batch(o.template_index(i),1);
        } else {
            const size_t max_count=*std::max_element(o.counts.begin(),o.counts.end());
            for (size_t offset=0; offset<max_count; offset+=o.batch_size) {
                for (size_t i=0; i<matrices.size(); ++i) {
                    if (offset>=o.counts[i]) continue;
                    add_batch(i,unsigned(std::min(size_t(o.batch_size),o.counts[i]-offset)));
                }
            }
        }
        const auto graph_start=Clock::now();
        if (o.graph) for (auto& t : tasks) t->prepare_graph();
        const double graph_setup_ms=o.graph ? elapsed(graph_start) : 0;
        const double setup_ms=elapsed(setup_start);
        auto loop = [&]() {
            for (auto& t : tasks) t->enqueue();
            for (auto& t : tasks) t->wait();
        };
        const auto warmup_start=Clock::now();
        for (size_t i=0; i<o.warmup; ++i) loop();
        const double warmup_ms=elapsed(warmup_start);
        std::vector<double> loop_ms(o.loops), submit_ms, drain_ms;
        if (o.diagnostics) { submit_ms.resize(o.loops); drain_ms.resize(o.loops); }
        for (size_t i=0; i<o.loops; ++i) {
            const auto start=Clock::now();
            if (o.diagnostics) {
                for (auto& t : tasks) t->enqueue();
                const auto submitted=Clock::now();
                for (auto& t : tasks) t->wait();
                const auto finished=Clock::now();
                submit_ms[i]=std::chrono::duration<double,std::milli>(submitted-start).count();
                drain_ms[i]=std::chrono::duration<double,std::milli>(finished-submitted).count();
                loop_ms[i]=std::chrono::duration<double,std::milli>(finished-start).count();
            } else {
                loop();
                loop_ms[i]=elapsed(start);
            }
        }
        const auto check_start=Clock::now();
        const double tolerance=sizeof(REAL)==8 ? 1e-10 : 1e-3;
        double max_residual=0;
        size_t checked_tasks=0;
        for (size_t i=0; i<tasks.size(); ++i) {
            try {
                tasks[i]->check();
                auto& m=*matrices[task_template[i]];
                for (unsigned copy=0; copy<tasks[i]->batch_size(); ++copy) {
                    const REAL* factors=tasks[i]->factors(copy);
                    if (o.check == "full") {
                        const double residual=m.symbolic->factor_residual(factors);
                        if (!std::isfinite(residual) || residual>tolerance) {
                            std::ostringstream message;
                            message << std::scientific << "factor residual " << residual
                                    << " exceeds tolerance " << tolerance;
                            throw std::runtime_error(message.str());
                        }
                        max_residual=std::max(max_residual,residual);
                    }
                    const auto x=m.symbolic->solve(&m.nics.state,std::vector<REAL>(m.symbolic->n,REAL(1)),factors);
                    for (REAL value : x) if (!std::isfinite(value)) throw std::runtime_error("nonfinite solution");
                    ++checked_tasks;
                }
            } catch (const std::exception& error) {
                throw std::runtime_error("batch " + std::to_string(i) + ": " + error.what());
            }
        }
        if (checked_tasks!=o.tasks) throw std::runtime_error("task coverage mismatch");
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
               << ",\n  \"schedule\": " << quoted(o.batch_size==1 ? "one-stream-per-task" : "one-stream-per-batch")
               << ", \"submission\": " << quoted(o.graph ? "single-host-thread-graph" : "single-host-thread-direct") << ','
               << "\n  \"tasks\": " << o.tasks << ", \"streams\": " << tasks.size()
               << ", \"warmup\": " << o.warmup << ", \"loops\": " << o.loops
               << ", \"requested_tmp_slots\": " << o.slots << ", \"batch_size\": " << o.batch_size
               << ",\n  \"task_device_bytes\": " << device_bytes << ", \"pinned_bytes\": " << pinned_bytes
               << ",\n  \"setup_ms\": " << setup_ms << ", \"validation_ms\": " << check_ms
               << ", \"graph\": " << (o.graph ? "true" : "false") << ", \"graph_setup_ms\": " << graph_setup_ms << ", \"warmup_ms\": " << warmup_ms
               << ", \"ms_per_loop\": " << total_ms/o.loops
               << ",\n  \"check_mode\": " << quoted(o.check)
               << ",\n  \"validation\": " << quoted(o.check == "full"
                   ? "all-tasks-final-loop-full-factor-residual-and-finite-solve"
                   : "all-tasks-final-loop-finite-factors-nonzero-diagonal-and-finite-solve")
               << ",\n  \"checked_tasks\": " << checked_tasks << ", \"max_factor_residual\": ";
        if (o.check == "full") report << max_residual; else report << "null";
        report << ", \"factor_residual_tolerance\": ";
        if (o.check == "full") report << tolerance; else report << "null";
        report << ",\n  \"diagnostics\": " << (o.diagnostics ? "true" : "false");
        if (o.diagnostics) {
            size_t launches=0, transfer_bytes=0;
            report << ",\n  \"template_work\": [";
            for (size_t i=0; i<matrices.size(); ++i) {
                const auto& a=*matrices[i]->symbolic;
                const unsigned slots=matrices[i]->gpu->tmp_slots();
                size_t batched=0, factor=0, update=0, clear=0;
                unsigned max_width=0, single_column_levels=0;
                for (size_t lev=0; lev+1<a.level_ptr.size(); ++lev) {
                    const unsigned width=unsigned(a.level_ptr[lev+1]-a.level_ptr[lev]);
                    max_width=std::max(max_width,width);
                    single_column_levels += width==1;
                    if (width>16) batched+=(size_t(width)+slots-1)/slots;
                    else for (int p=a.level_ptr[lev]; p<a.level_ptr[lev+1]; ++p) {
                        const unsigned col=a.level_idx[p];
                        ++factor; ++clear;
                        update += a.csr_r_ptr[col+1]-a.csr_diag_ptr[col]-1>0;
                    }
                }
                const size_t count=o.counts[i];
                add_bytes(launches,multiplied_bytes(template_batches[i],batched+factor+update+clear));
                add_bytes(transfer_bytes,multiplied_bytes(count,matrices[i]->gpu->task_host_bytes()));
                if (i) report << ',';
                report << "{\"levels\":" << a.num_lev << ",\"max_width\":" << max_width
                       << ",\"single_column_levels\":" << single_column_levels
                       << ",\"batched_launches\":" << batched << ",\"factor_launches\":" << factor
                       << ",\"update_launches\":" << update << ",\"clear_launches\":" << clear << '}';
            }
            report << "],\n  \"launches_per_loop\": " << launches
                   << ", \"h2d_bytes_per_loop\": " << transfer_bytes
                   << ", \"d2h_bytes_per_loop\": " << transfer_bytes;
            report << ",\n  \"submit_ms\": [";
            for (size_t i=0; i<o.loops; ++i) { if(i) report << ','; report << submit_ms[i]; }
            report << "],\n  \"drain_ms\": [";
            for (size_t i=0; i<o.loops; ++i) { if(i) report << ','; report << drain_ms[i]; }
            report << ']';
        }
        report << ",\n  \"ordering\": " << quoted(o.batch_size==1
            ? "logical-task-order" : "template-interleaved-batches");
        report << ",\n  \"batch_templates\": [";
        for (size_t i=0; i<tasks.size(); ++i) { if(i) report << ','; report << task_template[i]; }
        report << "],\n  \"batch_sizes\": [";
        for (size_t i=0; i<tasks.size(); ++i) { if(i) report << ','; report << tasks[i]->batch_size(); }
        report << ']';
        report << ",\n  \"task_templates\": [";
        for (size_t i=0; i<o.tasks; ++i) {
            if (i) report << ',';
            report << o.template_index(i);
        }
        report << ']';
        report << ",\n  \"templates\": [";
        for (size_t i=0; i<matrices.size(); ++i) {
            if (i) report << ',';
            const auto& m=*matrices[i];
            report << "{\"path\":" << quoted(o.inputs[i]) << ",\"n\":" << m.symbolic->n
                   << ",\"factor_nnz\":" << m.symbolic->nnz << ",\"tasks\":" << o.counts[i]
                   << ",\"batches\":" << template_batches[i] << ",\"effective_tmp_slots\":" << m.gpu->tmp_slots() << '}';
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
        std::cout.flush();
        if (!std::cout) throw std::runtime_error("failed to write standard output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
