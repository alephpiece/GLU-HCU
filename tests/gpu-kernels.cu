// Numeric kernel and host scheduling regressions, without preprocessing.
// Compile this translation unit instead of numeric.cu, and link symbolic.cc.
#include "gpu-compat.h"

namespace async_fault
{

#if defined(GLU_USE_HIP)
constexpr cudaError_t copy_error = hipErrorInvalidValue, sync_error = hipErrorUnknown;
#else
constexpr cudaError_t copy_error = cudaErrorInvalidValue, sync_error = cudaErrorUnknown;
#endif

struct State
{
    bool fail_upload, override_sync, same_stream = true;
    unsigned h2d_calls = 0, d2h_calls = 0, h2d_submitted = 0, sync_calls = 0, sync_successes = 0;
    cudaStream_t stream = nullptr;
};
State* active = nullptr;

struct Scope
{
    explicit Scope(State& state) { active = &state; }
    ~Scope() { active = nullptr; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

cudaError_t memcpy_async(void* destination,
    const void* source,
    size_t bytes,
    decltype(cudaMemcpyHostToDevice) kind,
    cudaStream_t stream)
{
    State* state = active;
    if (state && (kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToHost))
    {
        if (state->h2d_calls + state->d2h_calls == 0)
            state->stream = stream;
        else
            state->same_stream = state->same_stream && state->stream == stream;
        unsigned& calls = kind == cudaMemcpyHostToDevice ? state->h2d_calls : state->d2h_calls;
        ++calls;
        if (!state->fail_upload && kind == cudaMemcpyDeviceToHost && calls == 1)
            return copy_error;
    }
    const auto status = cudaMemcpyAsync(destination, source, bytes, kind, stream);
    if (state && kind == cudaMemcpyHostToDevice && status == cudaSuccess)
        ++state->h2d_submitted;
    // A failed upload must drain even if the batch DMA was already queued.
    if (state && state->fail_upload && kind == cudaMemcpyHostToDevice && status == cudaSuccess)
        return copy_error;
    return status;
}

cudaError_t stream_synchronize(cudaStream_t stream)
{
    const auto status = cudaStreamSynchronize(stream);
    if (active)
    {
        ++active->sync_calls;
        active->same_stream = active->same_stream && active->stream == stream;
        if (status == cudaSuccess)
        {
            ++active->sync_successes;
            if (active->override_sync)
                return sync_error;
        }
    }
    return status;
}

} // namespace async_fault

// Wrap only numeric.cu calls; keep runtime declarations and HIP aliases intact.
#pragma push_macro("cudaMemcpyAsync")
#pragma push_macro("cudaStreamSynchronize")
#undef cudaMemcpyAsync
#undef cudaStreamSynchronize
#define cudaMemcpyAsync async_fault::memcpy_async
#define cudaStreamSynchronize async_fault::stream_synchronize
#include "../src/numeric.cu"
#pragma pop_macro("cudaStreamSynchronize")
#pragma pop_macro("cudaMemcpyAsync")

#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

void check_gpu(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename T> class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t count) : pointer_(nullptr), count_(count)
    {
        check_gpu(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)), "allocate");
        check_gpu(cudaMemset(pointer_, 0, count * sizeof(T)), "initialize");
    }

    explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size())
    {
        check_gpu(cudaMemcpy(pointer_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }

    ~DeviceBuffer() { cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() { return pointer_; }
    std::vector<T> read() const
    {
        std::vector<T> host(count_);
        check_gpu(cudaMemcpy(host.data(), pointer_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return host;
    }

private:
    T* pointer_;
    size_t count_;
};

struct Stream
{
    cudaStream_t value;
    Stream() { check_gpu(cudaStreamCreate(&value), "create stream"); }
    ~Stream() { cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

struct Matrix
{
    unsigned n;
    std::vector<unsigned> column, row, diagonal;
    std::vector<unsigned> csr_row, csr_column, csr_diagonal;
    std::vector<int> sources;
    std::vector<REAL> values;
};

// The two source columns are independent, but update common destinations.
// Neighbouring logical groups have unequal loop lengths on a 64-lane wave.
// There are enough destinations to reuse every group even in a 1024-thread
// block. Prefix lengths place U(k,j) at lanes 0/31 and across chunk boundaries;
// long destinations also exercise the one-column kernel's 1024-element loop.
Matrix make_matrix(bool perturb)
{
    Matrix matrix;
    matrix.n = 1409;
    matrix.sources = {70, 73};
    std::vector<std::vector<std::pair<unsigned, REAL>>> columns(matrix.n);
    for (unsigned column = 0; column < matrix.n; ++column)
        columns[column].push_back(std::make_pair(column, REAL(4)));

    for (size_t i = 0; i < matrix.sources.size(); ++i)
    {
        const unsigned column = matrix.sources[i];
        columns[column][0].second = perturb ? (i ? REAL(-1e-12) : REAL(0)) : REAL(4 * (i + 1));
        for (unsigned row = 74; row < matrix.n; ++row)
        {
            if (row % 5 != 0)
                columns[column].push_back(std::make_pair(row, REAL(int(row % 9) - 4) / REAL(8)));
        }
    }

    const unsigned prefixes[] = {0, 1, 31, 32, 63, 64, 70};
    const unsigned tails[] = {0, 1, 7, 30, 31, 32, 33, 63, 65, 96, 1025, 1100};
    for (unsigned target = 0; target < 67; ++target)
    {
        const unsigned column = 1000 + target;
        std::vector<unsigned> rows;
        for (unsigned row = 0; row < prefixes[target % 7]; ++row)
            rows.push_back(row);
        rows.push_back(70);
        rows.push_back(73);
        for (unsigned row = 74; row < 74 + tails[target % 12]; ++row)
            rows.push_back(row);
        rows.push_back(column);
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        columns[column].clear();
        for (unsigned row : rows)
        {
            const REAL value = REAL(0.5) + REAL(column % 11) / REAL(16) + REAL(row % 13) / REAL(32);
            columns[column].push_back(std::make_pair(row, value));
        }
    }

    matrix.column.push_back(0);
    matrix.diagonal.resize(matrix.n);
    std::vector<std::vector<unsigned>> rows(matrix.n);
    for (unsigned column = 0; column < matrix.n; ++column)
    {
        for (const auto& entry : columns[column])
        {
            if (entry.first == column)
                matrix.diagonal[column] = matrix.row.size();
            matrix.row.push_back(entry.first);
            matrix.values.push_back(entry.second);
            rows[entry.first].push_back(column);
        }
        matrix.column.push_back(matrix.row.size());
    }
    matrix.csr_row.push_back(0);
    matrix.csr_diagonal.resize(matrix.n);
    for (unsigned row = 0; row < matrix.n; ++row)
    {
        for (unsigned column : rows[row])
        {
            if (column == row)
                matrix.csr_diagonal[row] = matrix.csr_column.size();
            matrix.csr_column.push_back(column);
        }
        matrix.csr_row.push_back(matrix.csr_column.size());
    }
    return matrix;
}

Matrix make_empty_lower_matrix()
{
    Matrix matrix;
    matrix.n = 2;
    matrix.column = {0, 1, 3};
    matrix.row = {0, 0, 1};
    matrix.diagonal = {0, 2};
    matrix.csr_row = {0, 2, 3};
    matrix.csr_column = {0, 1, 1};
    matrix.csr_diagonal = {0, 2};
    matrix.sources = {0};
    matrix.values = {REAL(0), REAL(3), REAL(4)};
    return matrix;
}

std::vector<REAL> reference_update(const Matrix& matrix, bool perturb, REAL threshold)
{
    std::vector<REAL> expected(matrix.values);
    for (int source : matrix.sources)
    {
        const unsigned diagonal = matrix.diagonal[source];
        if (perturb && std::fabs(expected[diagonal]) < threshold)
            expected[diagonal] = threshold;
        std::vector<REAL> lower(matrix.n, REAL(0));
        for (unsigned p = diagonal + 1; p < matrix.column[source + 1]; ++p)
        {
            expected[p] /= expected[diagonal];
            lower[matrix.row[p]] = expected[p];
        }
        for (unsigned q = matrix.csr_diagonal[source] + 1; q < matrix.csr_row[source + 1]; ++q)
        {
            const unsigned column = matrix.csr_column[q];
            REAL upper = 0;
            for (unsigned p = matrix.column[column]; p < matrix.column[column + 1]; ++p)
                if (matrix.row[p] == unsigned(source))
                    upper = matrix.values[p];
            for (unsigned p = matrix.column[column]; p < matrix.column[column + 1]; ++p)
                if (matrix.row[p] > unsigned(source))
                    expected[p] -= lower[matrix.row[p]] * upper;
        }
    }
    return expected;
}

struct DeviceMatrix
{
    DeviceBuffer<unsigned> column, row, diagonal, csr_row, csr_column, csr_diagonal;
    DeviceBuffer<int> sources;
    DeviceBuffer<REAL> values, temporary;

    explicit DeviceMatrix(const Matrix& matrix)
        : column(matrix.column), row(matrix.row), diagonal(matrix.diagonal), csr_row(matrix.csr_row),
          csr_column(matrix.csr_column), csr_diagonal(matrix.csr_diagonal), sources(matrix.sources),
          values(matrix.values), temporary(size_t(matrix.n) * matrix.sources.size())
    {
        check_gpu(cudaDeviceSynchronize(), "initialize test data");
    }
};

void compare(const std::vector<REAL>& actual, const std::vector<REAL>& expected, const char* field)
{
    if (actual.size() != expected.size())
        throw std::runtime_error(std::string(field) + " size mismatch");
    for (size_t i = 0; i < actual.size(); ++i)
    {
        const REAL tolerance =
            REAL(64) * std::numeric_limits<REAL>::epsilon() * std::max(REAL(1), std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance)
        {
            std::ostringstream error;
            error.precision(std::numeric_limits<REAL>::max_digits10);
            error << field << '[' << i << "]: expected " << expected[i] << ", got " << actual[i];
            throw std::runtime_error(error.str());
        }
    }
}

void verify(DeviceMatrix& device, const Matrix& matrix, bool perturb, REAL threshold)
{
    check_gpu(cudaDeviceSynchronize(), "execute kernels");
    compare(device.values.read(), reference_update(matrix, perturb, threshold), "LU");
    const std::vector<REAL> temporary = device.temporary.read();
    // Clearing scratch is required for its reuse by the next level/chunk.
    for (REAL value : temporary)
        if (value != REAL(0))
            throw std::runtime_error("temporary storage was not cleared");
}

void test_batched(const Matrix& matrix, unsigned threads, bool perturb, const std::string& name)
{
    const REAL threshold = REAL(0.5);
    DeviceMatrix device(matrix);
    const unsigned blocks = matrix.sources.size();
    if (perturb)
        RL<UpdateMode::Scan, true><<<blocks, threads>>>(device.column.data(), device.row.data(), device.values.data(),
            device.diagonal.data(), device.csr_row.data(), device.csr_column.data(), device.csr_diagonal.data(),
            device.sources.data(), device.temporary.data(), matrix.n, 0, 0, 0, 0, threshold);
    else
        RL<UpdateMode::Scan><<<blocks, threads>>>(device.column.data(), device.row.data(), device.values.data(),
            device.diagonal.data(), device.csr_row.data(), device.csr_column.data(), device.csr_diagonal.data(),
            device.sources.data(), device.temporary.data(), matrix.n, 0, 0);
    check_gpu(cudaGetLastError(), "launch batched kernel");
    verify(device, matrix, perturb, threshold);
    std::cout << "PASS " << name << '\n';
}

void test_one_column(const Matrix& matrix, bool perturb, const std::string& name, unsigned threads = 1024)
{
    const REAL threshold = REAL(0.5);
    DeviceMatrix device(matrix);
    Stream streams[2];
    for (size_t slot = 0; slot < matrix.sources.size(); ++slot)
    {
        const unsigned source = matrix.sources[slot];
        if (perturb)
            RL_onecol_factorizeCurrentCol<true><<<1, threads, 0, streams[slot].value>>>(device.column.data(),
                device.row.data(), device.values.data(), device.diagonal.data(), source, device.temporary.data(), slot,
                matrix.n, 0, 0, threshold);
        else
            RL_onecol_factorizeCurrentCol<><<<1, threads, 0, streams[slot].value>>>(device.column.data(),
                device.row.data(), device.values.data(), device.diagonal.data(), source, device.temporary.data(), slot,
                matrix.n);
        check_gpu(cudaGetLastError(), "launch factor kernel");
        const unsigned targets = matrix.csr_row[source + 1] - matrix.csr_diagonal[source] - 1;
        if (targets)
        {
            RL_onecol_updateSubmat<UpdateMode::Scan><<<targets, threads, 0, streams[slot].value>>>(device.column.data(),
                device.row.data(), device.values.data(), device.csr_column.data(), device.csr_diagonal.data(), source,
                device.temporary.data(), slot, matrix.n);
            check_gpu(cudaGetLastError(), "launch update kernel");
        }
        RL_onecol_cleartmpMem<<<1, threads, 0, streams[slot].value>>>(device.column.data(), device.row.data(),
            device.diagonal.data(), source, device.temporary.data(), slot, matrix.n);
        check_gpu(cudaGetLastError(), "launch clear kernel");
    }
    verify(device, matrix, perturb, threshold);
    std::cout << "PASS " << name << '\n';
}

void test_limited_slots()
{
    // Small levels use one scratch slot, even when a later level is wider.
    // The next level has three columns with nonempty L parts, all updating the
    // final diagonal. Reusing slot zero must serialize factor/update/clear.
    std::vector<unsigned> column{0, 5, 7, 9, 11, 15};
    std::vector<unsigned> row{0, 1, 2, 3, 4, 1, 4, 2, 4, 3, 4, 1, 2, 3, 4};
    std::vector<double> values{4, 1, 2, 3, 1, 4, 1, 4, 2, 4, 3, 1, 2, 3, 8};
    const std::vector<REAL> expected{4, .25, .5, .75, .25, 4, .25, 4, .5, 4, .75, 1, 2, 3, 4.5};
    std::ostringstream out, err;
    Symbolic_Matrix matrix(5, out, err);
    matrix.fill_in(row.data(), column.data());
    matrix.csr();
    matrix.predictLU(row.data(), column.data(), values.data());
    matrix.leveling();
    if (matrix.level_ptr != std::vector<int>{0, 1, 4, 5})
        throw std::runtime_error("limited-slot fixture has unexpected levels");
    if (!LUonDevice(matrix, out, err, false))
        throw std::runtime_error("limited-slot factorization failed: " + err.str());
    compare(matrix.val, expected, "limited-slot LU");
    std::cout << "PASS host/one-slot-three-columns\n";
}

void verify_template_bytes(const GpuMatrix& gpu,
    const Symbolic_Matrix& matrix,
    UpdateMode mode,
    SnapshotPolicy snapshot = SnapshotPolicy::PerCopy)
{
    const size_t structure =
        sizeof(unsigned) *
            (matrix.sym_c_ptr.size() + matrix.l_col_ptr.size() + matrix.csr_r_ptr.size() + matrix.csr_diag_ptr.size() +
                (mode == UpdateMode::Scan ? matrix.sym_r_idx.size() + matrix.csr_c_idx.size() : 0)) +
        sizeof(int) * matrix.level_idx.size();
    uint64_t updates = 0;
    for (unsigned c = 0; c < matrix.n; ++c)
        updates += uint64_t(matrix.sym_c_ptr[c + 1] - matrix.l_col_ptr[c] - 1) *
                   (matrix.csr_r_ptr[c + 1] - matrix.csr_diag_ptr[c] - 1);
    const size_t map = mode == UpdateMode::Mapped
                           ? sizeof(uint64_t) * (size_t(matrix.n) + 1) + sizeof(unsigned) * matrix.csr_c_idx.size() +
                                 size_t(updates) * sizeof(uint16_t)
                           : 0;
    const size_t task =
        sizeof(REAL) * (matrix.val.size() + (mode == UpdateMode::Scan ? size_t(matrix.n) * gpu.tmp_slots() : 0));
    if (gpu.task_device_bytes() != task)
        throw std::runtime_error("incorrect task device byte accounting");
    size_t extra = 0;
    if (mode == UpdateMode::Mapped)
    {
        extra = sizeof(uint64_t) * (matrix.level_idx.size() + 1);
        for (size_t lev = 0; lev + 1 < matrix.level_ptr.size(); ++lev)
            if (matrix.level_ptr[lev + 1] - matrix.level_ptr[lev] <= 64)
                extra += sizeof(int) * size_t(matrix.level_ptr[lev + 1] - matrix.level_ptr[lev]);
    }
    for (unsigned copies : {1u, 2u, 3u, 31u, 32u, 33u, 42u})
    {
        const unsigned lanes = 1u << filled_copy_bits(copies, 5);
        const unsigned pitch = mode == UpdateMode::Mapped && copies > 1 ? (copies + lanes - 1) / lanes * lanes : 0;
        const size_t expected =
            sizeof(REAL) * (matrix.val.size() * (copies + pitch) +
                               (mode == UpdateMode::Scan ? size_t(matrix.n) * gpu.tmp_slots() * copies : 0));
        if (gpu.task_device_bytes(copies) != expected)
            throw std::runtime_error("incorrect batch device bytes");
    }
    if (gpu.update_map_bytes() != map ||
        gpu.shared_device_bytes() !=
            structure + map + extra + (snapshot == SnapshotPolicy::Device ? matrix.val.size() * sizeof(REAL) : 0))
        throw std::runtime_error("incorrect template device byte accounting");
    if (!std::isfinite(gpu.update_map_setup_ms()) || gpu.update_map_setup_ms() < 0 ||
        (mode == UpdateMode::Scan && gpu.update_map_setup_ms() != 0))
        throw std::runtime_error("incorrect map setup timing");
}

void test_repeated_tasks(bool graph = false,
    unsigned copies = 1,
    UpdateMode mode = UpdateMode::Scan,
    SnapshotPolicy snapshot = SnapshotPolicy::PerCopy)
{
    std::ostringstream out, err;
    std::vector<std::unique_ptr<GpuTask>> tasks;
    std::vector<std::vector<REAL>> expected;
    for (unsigned variant = 0; variant < 2; ++variant)
    {
        // 34 independent columns, then one dependent column. Exercises batched
        // chunking with fewer slots than the level width, then the small path.
        const unsigned n = 35;
        const double diagonal = 4 + variant;
        std::vector<unsigned> col{0}, row;
        std::vector<double> val;
        std::vector<REAL> factors;
        for (unsigned j = 0; j < n - 1; ++j)
        {
            row.push_back(j);
            val.push_back(diagonal);
            factors.push_back(REAL(diagonal));
            row.push_back(n - 1);
            val.push_back(1);
            factors.push_back(REAL(1 / diagonal));
            col.push_back(unsigned(row.size()));
        }
        for (unsigned i = 0; i < n; ++i)
        {
            row.push_back(i);
            val.push_back(i == n - 1 ? 50 : 1);
            factors.push_back(i == n - 1 ? REAL(50 - (n - 1) / diagonal) : REAL(1));
        }
        col.push_back(unsigned(row.size()));
        Symbolic_Matrix a(n, out, err);
        a.fill_in(row.data(), col.data());
        a.csr();
        a.predictLU(row.data(), col.data(), val.data());
        a.leveling();
        GpuMatrix matrix(a, variant ? 3 : 1, false, mode, snapshot);
        verify_template_bytes(matrix, a, mode, snapshot);
        const unsigned expected_slots = mode == UpdateMode::Scan ? (variant ? 3u : 1u) : 0;
        if (matrix.tmp_slots() != expected_slots)
            throw std::runtime_error("wrong effective scratch slots");
        for (unsigned copy = 0; copy < 2; ++copy)
        {
            tasks.emplace_back(new GpuTask(matrix, copies));
            if (graph)
            {
                tasks.back()->prepare_graph();
                tasks.back()->prepare_graph();
            }
            expected.push_back(factors);
        }
        // Both the host matrix and GpuMatrix wrapper die here. Tasks must retain
        // all input, structure and launch metadata required by later enqueues.
    }
    for (unsigned loop = 0; loop < 4; ++loop)
    {
        if (loop % 2)
            for (auto& task : tasks)
            {
                task->upload_input();
                bool rejected_read = false, rejected_requeue = false, rejected_capture = false, rejected_wait = false;
                try
                {
                    task->factors();
                }
                catch (const std::logic_error&)
                {
                    rejected_read = true;
                }
                try
                {
                    task->upload_input();
                }
                catch (const std::logic_error&)
                {
                    rejected_requeue = true;
                }
                try
                {
                    task->prepare_graph();
                }
                catch (const std::logic_error&)
                {
                    rejected_capture = true;
                }
                try
                {
                    task->wait();
                }
                catch (const std::logic_error&)
                {
                    rejected_wait = true;
                }
                if (!rejected_read || !rejected_requeue || !rejected_capture || !rejected_wait)
                    throw std::runtime_error("queued input allowed invalid reuse or result access");
            }
        for (auto& task : tasks)
            task->enqueue();
        if (loop == 0)
        {
            bool rejected_read = false, rejected_reuse = false, rejected_capture = false;
            try
            {
                tasks[0]->factors();
            }
            catch (const std::logic_error&)
            {
                rejected_read = true;
            }
            try
            {
                tasks[0]->enqueue();
            }
            catch (const std::logic_error&)
            {
                rejected_reuse = true;
            }
            try
            {
                tasks[0]->prepare_graph();
            }
            catch (const std::logic_error&)
            {
                rejected_capture = true;
            }
            if (!rejected_read || !rejected_reuse || !rejected_capture)
                throw std::runtime_error("pending task allowed read or reuse");
        }
        for (auto& task : tasks)
            task->wait();
        for (size_t i = 0; i < tasks.size(); ++i)
        {
            tasks[i]->check();
            for (unsigned copy = 0; copy < copies; ++copy)
                compare(std::vector<REAL>(tasks[i]->factors(copy), tasks[i]->factors(copy) + expected[i].size()),
                    expected[i], "repeated independent tasks");
        }
    }
    std::cout << "PASS host/repeated-mixed-tasks-and-template-lifetime/"
              << (mode == UpdateMode::Scan ? "scan" : "mapped") << '\n';
}

void test_repeated_mapped_indices()
{
    // Repeated off-diagonal entries still refer to the first matching slot in
    // the target column, both when reading U(k,j) and when writing the update at (i,j).
    for (bool repeated_pivot : {false, true})
    {
        std::vector<unsigned> rows = repeated_pivot ? std::vector<unsigned>{0, 2, 0, 0, 1, 2, 2}
                                                    : std::vector<unsigned>{0, 2, 2, 0, 1, 2, 2, 2};
        std::vector<unsigned> columns = repeated_pivot ? std::vector<unsigned>{0, 2, 6, 7}
                                                       : std::vector<unsigned>{0, 3, 7, 8};
        std::ostringstream out, err;
        Symbolic_Matrix a(3, out, err);
        a.fill_in(rows.data(), columns.data());
        a.csr();
        const std::vector<uint64_t> offsets{0, 2, 2, 2};
        std::vector<unsigned> host_destinations(2);
        const auto pivots = mapped_pivot_slots(a, offsets, &host_destinations);
        Buffer<uint64_t> bases;
        Buffer<unsigned> col, diag, csr_diag, pivot_slots, output;
        bases.upload(offsets);
        col.upload(a.sym_c_ptr);
        diag.upload(a.l_col_ptr);
        csr_diag.upload(a.csr_diag_ptr);
        pivot_slots.upload(pivots);
        create_mapped_destinations(a, 2, 1, bases.ptr, col.ptr, diag.ptr, csr_diag.ptr, pivot_slots.ptr, output);
        std::vector<unsigned> destinations(2);
        gpu_check(cudaMemcpy(destinations.data(), output.ptr, output.bytes, cudaMemcpyDeviceToHost), "read map");
        const unsigned first = a.sym_c_ptr[1];
        if (destinations != host_destinations || destinations != std::vector<unsigned>{5, 5} || pivots[1] != first ||
            (repeated_pivot && pivots[2] != first))
            throw std::runtime_error("mapped lookup changed a repeated factor slot");

        std::vector<uint16_t> host_delta(2);
        if (mapped_pivot_slots(a, offsets, &host_delta) != pivots)
            throw std::runtime_error("relative map changed pivot slots");
        Buffer<uint16_t> relative;
        create_mapped_destinations(a, 2, 1, bases.ptr, col.ptr, diag.ptr, csr_diag.ptr, pivot_slots.ptr, relative);
        std::vector<uint16_t> delta(2);
        gpu_check(cudaMemcpy(delta.data(), relative.ptr, relative.bytes, cudaMemcpyDeviceToHost),
            "read relative duplicate map");
        if (delta != host_delta || pivots[1] + delta[0] != 5 || pivots[repeated_pivot ? 2 : 1] + delta[1] != 5)
            throw std::runtime_error("relative slots changed first-occurrence lookup");
        // Remove the target row i after recording the slots for U(k,j).
        // The device map constructor must report the missing entry.
        std::fill(a.sym_r_idx.begin() + a.l_col_ptr[1] + 1, a.sym_r_idx.begin() + a.sym_c_ptr[2], 1u);
        Buffer<unsigned> invalid_output;
        try
        {
            create_mapped_destinations(a, 2, 1, bases.ptr, col.ptr, diag.ptr, csr_diag.ptr, pivot_slots.ptr,
                invalid_output);
        }
        catch (const std::runtime_error& error)
        {
            if (std::string(error.what()) != "update map missing lower row")
                throw;
            continue;
        }
        throw std::runtime_error("device map accepted a missing destination");
    }
}

void test_zero_update_map()
{
    unsigned column[]{0, 1, 2}, row[]{0, 1};
    double values[]{2, 4};
    std::ostringstream out, err;
    Symbolic_Matrix matrix(2, out, err);
    matrix.fill_in(row, column);
    matrix.csr();
    matrix.predictLU(row, column, values);
    matrix.leveling();
    GpuMatrix gpu(matrix, 1, false, UpdateMode::Mapped);
    verify_template_bytes(gpu, matrix, UpdateMode::Mapped);
    GpuTask task(gpu);
    task.prepare_graph();
    task.enqueue();
    task.wait();
    task.check();
    compare(std::vector<REAL>(task.factors(), task.factors() + 2), {REAL(2), REAL(4)}, "empty update map");
    std::cout << "PASS host/mapped-zero-update-entries\n";
}

void test_distinct_copies(bool one_column, bool perturb)
{
    const unsigned n = 2, copies = 3, slots = 2, nnz = 4;
    DeviceBuffer<unsigned> col(std::vector<unsigned>{0, 2, 4}), row(std::vector<unsigned>{0, 1, 0, 1});
    DeviceBuffer<unsigned> diag(std::vector<unsigned>{0, 3}), csr_row(std::vector<unsigned>{0, 2, 4});
    DeviceBuffer<unsigned> csr_col(std::vector<unsigned>{0, 1, 0, 1}), csr_diag(std::vector<unsigned>{0, 3});
    DeviceBuffer<int> levels(std::vector<int>{0});
    std::vector<REAL> input, expected;
    for (unsigned i = 0; i < copies; ++i)
    {
        const REAL a = perturb ? REAL(0) : REAL(i + 2), b = REAL(i + 5), d = REAL(i + 10);
        const REAL pivot = perturb ? REAL(2) : a;
        input.insert(input.end(), {a, REAL(i + 1), b, d});
        expected.insert(expected.end(), {pivot, REAL(i + 1) / pivot, b, d - REAL(i + 1) / pivot * b});
    }
    DeviceBuffer<REAL> values(input), scratch(n * slots * copies);
    Stream stream;
    if (one_column)
    {
        if (perturb)
            RL_onecol_factorizeCurrentCol<true><<<dim3(1, copies), 1024, 0, stream.value>>>(col.data(), row.data(),
                values.data(), diag.data(), 0, scratch.data(), 1, n, nnz, n * slots, REAL(2));
        else
            RL_onecol_factorizeCurrentCol<><<<dim3(1, copies), 1024, 0, stream.value>>>(col.data(), row.data(),
                values.data(), diag.data(), 0, scratch.data(), 1, n, nnz, n * slots);
        RL_onecol_updateSubmat<UpdateMode::Scan><<<dim3(1, copies), 1024, 0, stream.value>>>(col.data(), row.data(),
            values.data(), csr_col.data(), csr_diag.data(), 0, scratch.data(), 1, n, nnz, n * slots);
        RL_onecol_cleartmpMem<<<dim3(1, copies), 1024, 0, stream.value>>>(col.data(), row.data(), diag.data(), 0,
            scratch.data(), 1, n, n * slots);
    }
    else if (perturb)
        RL<UpdateMode::Scan, true><<<dim3(1, copies), 64, 0, stream.value>>>(col.data(), row.data(), values.data(),
            diag.data(), csr_row.data(), csr_col.data(), csr_diag.data(), levels.data(), scratch.data(), n, 0, 0, nnz,
            n * slots, REAL(2));
    else
        RL<UpdateMode::Scan><<<dim3(1, copies), 64, 0, stream.value>>>(col.data(), row.data(), values.data(),
            diag.data(), csr_row.data(), csr_col.data(), csr_diag.data(), levels.data(), scratch.data(), n, 0, 0, nnz,
            n * slots);
    check_gpu(cudaGetLastError(), "launch distinct-copy kernels");
    check_gpu(cudaStreamSynchronize(stream.value), "wait distinct copies");
    compare(values.read(), expected, "distinct batch values");
    compare(scratch.read(), std::vector<REAL>(n * slots * copies, REAL(0)), "distinct batch scratch cleanup");
    std::cout << "PASS distinct batch copies " << one_column << ' ' << perturb << '\n';
}

// Unlike make_matrix(), this pattern contains every source L row in each of
// its targets. Sources are independent and have unequal L lengths (65/33/1/0),
// but update overlapping target entries. The empty source still has targets.
Matrix make_mapped_matrix(bool perturb, unsigned copy)
{
    Matrix matrix;
    matrix.n = 100;
    matrix.sources = {2, 5, 8, 11};
    const unsigned lengths[] = {65, 33, 1, 0};
    const unsigned targets[] = {5, 3, 4, 2};
    std::vector<std::vector<std::pair<unsigned, REAL>>> columns(matrix.n);
    for (unsigned c = 0; c < matrix.n; ++c)
        columns[c].emplace_back(c, REAL(4));
    for (unsigned s = 0; s < matrix.sources.size(); ++s)
    {
        const unsigned c = matrix.sources[s];
        columns[c][0].second = perturb ? (s % 2 ? REAL(-1e-12) : REAL(0)) : REAL(2u << (s + copy));
        for (unsigned i = 0; i < lengths[s]; ++i)
            columns[c].emplace_back(20 + i, REAL(int(i % 7) - 3) * REAL(copy + 1) / REAL(8));
    }
    for (unsigned t = 0; t < 5; ++t)
    {
        const unsigned c = 90 + t;
        std::vector<unsigned> rows{0, 6, 87, c};
        for (unsigned s = 0; s < matrix.sources.size(); ++s)
        {
            if (t >= targets[s])
                continue;
            rows.push_back(matrix.sources[s]);
            for (unsigned i = 0; i < lengths[s]; ++i)
                rows.push_back(20 + i);
        }
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        columns[c].clear();
        for (unsigned row : rows)
            columns[c].emplace_back(row,
                REAL(0.5) + REAL(row % 7) / REAL(16) + REAL(t) / REAL(32) + REAL(copy) / REAL(8));
    }
    matrix.column.push_back(0);
    matrix.diagonal.resize(matrix.n);
    std::vector<std::vector<unsigned>> rows(matrix.n);
    for (unsigned c = 0; c < matrix.n; ++c)
    {
        for (const auto& entry : columns[c])
        {
            if (entry.first == c)
                matrix.diagonal[c] = matrix.row.size();
            matrix.row.push_back(entry.first);
            matrix.values.push_back(entry.second);
            rows[entry.first].push_back(c);
        }
        matrix.column.push_back(matrix.row.size());
    }
    matrix.csr_row.push_back(0);
    matrix.csr_diagonal.resize(matrix.n);
    for (unsigned row = 0; row < matrix.n; ++row)
    {
        for (unsigned c : rows[row])
        {
            if (c == row)
                matrix.csr_diagonal[row] = matrix.csr_column.size();
            matrix.csr_column.push_back(c);
        }
        matrix.csr_row.push_back(matrix.csr_column.size());
    }
    return matrix;
}

void test_mapped(bool one_column, bool perturb, unsigned threads)
{
    const unsigned copies = 3, slots = 2;
    const REAL threshold = REAL(0.5);
    const Matrix matrix = make_mapped_matrix(perturb, 0);
    const size_t nnz = matrix.values.size(), stride = size_t(matrix.n) * slots;
    // Use a dense coordinate lookup as an independent map oracle, rather than
    // duplicating the production builder's sorted-column merge.
    const unsigned missing = std::numeric_limits<unsigned>::max();
    std::vector<unsigned> positions(size_t(matrix.n) * matrix.n, missing);
    for (unsigned c = 0; c < matrix.n; ++c)
        for (unsigned p = matrix.column[c]; p < matrix.column[c + 1]; ++p)
            positions[size_t(c) * matrix.n + matrix.row[p]] = p;
    std::vector<uint64_t> bases(matrix.n + 1);
    std::vector<unsigned> destinations, pivots(matrix.csr_column.size(), missing);
    for (unsigned c = 0; c < matrix.n; ++c)
    {
        bases[c] = destinations.size();
        for (unsigned q = matrix.csr_diagonal[c] + 1; q < matrix.csr_row[c + 1]; ++q)
        {
            const unsigned target = matrix.csr_column[q];
            pivots[q] = positions[size_t(target) * matrix.n + c];
            if (pivots[q] == missing)
                throw std::runtime_error("mapped fixture missing U entry");
            for (unsigned p = matrix.diagonal[c] + 1; p < matrix.column[c + 1]; ++p)
            {
                const unsigned destination = positions[size_t(target) * matrix.n + matrix.row[p]];
                if (destination == missing)
                    throw std::runtime_error("mapped fixture is not closed");
                destinations.push_back(destination);
            }
        }
    }
    bases[matrix.n] = destinations.size();
    std::vector<REAL> input, expected;
    for (unsigned copy = 0; copy < copies; ++copy)
    {
        const Matrix distinct = make_mapped_matrix(perturb, copy);
        input.insert(input.end(), distinct.values.begin(), distinct.values.end());
        const auto result = reference_update(distinct, perturb, threshold);
        expected.insert(expected.end(), result.begin(), result.end());
    }
    DeviceBuffer<unsigned> column(matrix.column), row(matrix.row), diagonal(matrix.diagonal);
    DeviceBuffer<unsigned> csr_row(matrix.csr_row), csr_column(matrix.csr_column), csr_diagonal(matrix.csr_diagonal);
    DeviceBuffer<uint64_t> map_bases(bases);
    DeviceBuffer<unsigned> map_destinations(destinations), map_pivots(pivots);
    // Nonzero levelHead plus a second chunk exercises both source-list offsets.
    DeviceBuffer<int> sources(std::vector<int>{0, 2, 5, 8, 11});
    DeviceBuffer<REAL> values(input);
    Stream stream;
    for (unsigned repeat = 0; repeat < 2; ++repeat)
    {
        // No scratch allocation: repeated mapped execution only restores values.
        check_gpu(cudaMemcpy(values.data(), input.data(), input.size() * sizeof(REAL), cudaMemcpyHostToDevice),
            "restore mapped inputs");
        if (one_column)
        {
            for (int c : matrix.sources)
            {
                if (perturb)
                    RL_onecol_factorizeCurrentCol<true, false>
                        <<<dim3(1, copies), threads, 0, stream.value>>>(column.data(), row.data(), values.data(),
                            diagonal.data(), c, nullptr, 0, matrix.n, nnz, stride, threshold);
                else
                    RL_onecol_factorizeCurrentCol<false, false>
                        <<<dim3(1, copies), threads, 0, stream.value>>>(column.data(), row.data(), values.data(),
                            diagonal.data(), c, nullptr, 0, matrix.n, nnz, stride);
                check_gpu(cudaGetLastError(), "launch mapped source normalization");
                const unsigned targets = matrix.csr_row[c + 1] - matrix.csr_diagonal[c] - 1;
                RL_onecol_updateSubmat<UpdateMode::Mapped>
                    <<<dim3(targets, copies), threads, 0, stream.value>>>(column.data(), row.data(), values.data(),
                        csr_column.data(), csr_diagonal.data(), c, nullptr, 0, matrix.n, nnz, stride, diagonal.data(),
                        map_bases.data(), map_destinations.data(), map_pivots.data());
                check_gpu(cudaGetLastError(), "launch mapped column update");
            }
        }
        else
        {
            for (unsigned offset : {0u, 2u})
            {
                if (perturb)
                    RL<UpdateMode::Mapped, true><<<dim3(slots, copies), threads, 0, stream.value>>>(column.data(),
                        row.data(), values.data(), diagonal.data(), csr_row.data(), csr_column.data(),
                        csr_diagonal.data(), sources.data(), nullptr, matrix.n, 1, offset, nnz, stride, threshold,
                        map_bases.data(), map_destinations.data(), map_pivots.data());
                else
                    RL<UpdateMode::Mapped><<<dim3(slots, copies), threads, 0, stream.value>>>(column.data(), row.data(),
                        values.data(), diagonal.data(), csr_row.data(), csr_column.data(), csr_diagonal.data(),
                        sources.data(), nullptr, matrix.n, 1, offset, nnz, stride, 0, map_bases.data(),
                        map_destinations.data(), map_pivots.data());
                check_gpu(cudaGetLastError(), "launch mapped level chunk");
            }
        }
        check_gpu(cudaStreamSynchronize(stream.value), "wait mapped kernels");
        compare(values.read(), expected, "mapped factors and distinct copies");
    }
    std::cout << "PASS mapped/" << (one_column ? "one-column/" : "RL/") << (perturb ? "perturb/" : "plain/") << threads
              << '\n';
}

Matrix make_layout_matrix(bool perturb, unsigned copy, unsigned variant = 0)
{
    Matrix matrix;
    matrix.n = 100;
    matrix.sources = {2, 5, 8, 11};
    const unsigned shapes[][4] = {{65, 33, 1, 0}, {0, 1, 2, 3}, {17, 31, 32, 33}, {9, 3, 2, 0}, {7, 8, 9, 0}};
    const unsigned fanouts[][4] = {{5, 3, 4, 2}, {5, 1, 0, 2}, {5, 3, 4, 2}, {5, 5, 5, 2}, {5, 3, 0, 2}};
    if (variant >= 5)
        throw std::invalid_argument("unknown mapped fixture");
    const unsigned* lengths = shapes[variant];
    const unsigned* targets = fanouts[variant];
    std::vector<std::vector<std::pair<unsigned, REAL>>> columns(matrix.n);
    for (unsigned c = 0; c < matrix.n; ++c)
        columns[c].emplace_back(c, REAL(4));
    for (unsigned s = 0; s < matrix.sources.size(); ++s)
    {
        const unsigned c = matrix.sources[s];
        columns[c][0].second =
            perturb ? (s % 2 ? REAL(-1e-12) : REAL(0)) : REAL(2u << s) * (REAL(1) + REAL(copy) / REAL(4));
        for (unsigned i = 0; i < lengths[s]; ++i)
            columns[c].emplace_back(20 + i, REAL(int(i % 7) - 3) * REAL(copy + 1) / REAL(8));
    }
    for (unsigned t = 0; t < 5; ++t)
    {
        const unsigned c = 90 + t;
        std::vector<unsigned> rows{0, 6, 87, c};
        for (unsigned s = 0; s < matrix.sources.size(); ++s)
        {
            if (t >= targets[s])
                continue;
            rows.push_back(matrix.sources[s]);
            for (unsigned i = 0; i < lengths[s]; ++i)
                rows.push_back(20 + i);
        }
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        columns[c].clear();
        for (unsigned row : rows)
            columns[c].emplace_back(row,
                REAL(0.5) + REAL(row % 7) / REAL(16) + REAL(t) / REAL(32) + REAL(copy) / REAL(8));
    }
    matrix.column.push_back(0);
    matrix.diagonal.resize(matrix.n);
    std::vector<std::vector<unsigned>> rows(matrix.n);
    for (unsigned c = 0; c < matrix.n; ++c)
    {
        for (const auto& entry : columns[c])
        {
            if (entry.first == c)
                matrix.diagonal[c] = matrix.row.size();
            matrix.row.push_back(entry.first);
            matrix.values.push_back(entry.second);
            rows[entry.first].push_back(c);
        }
        matrix.column.push_back(matrix.row.size());
    }
    matrix.csr_row.push_back(0);
    matrix.csr_diagonal.resize(matrix.n);
    for (unsigned row = 0; row < matrix.n; ++row)
    {
        for (unsigned c : rows[row])
        {
            if (c == row)
                matrix.csr_diagonal[row] = matrix.csr_column.size();
            matrix.csr_column.push_back(c);
        }
        matrix.csr_row.push_back(matrix.csr_column.size());
    }
    return matrix;
}

void test_layout(unsigned copies, unsigned variant, bool perturb, unsigned path)
{
    const bool soa = path != 0, normalized = path == 2;
    const unsigned bits = filled_copy_bits(copies, 5), lanes = 1u << bits;
    const unsigned alignment = 1u << filled_copy_bits(copies, 5);
    const unsigned pitch = (copies + alignment - 1) / alignment * alignment, tiles_y = (copies + lanes - 1) / lanes;
    const REAL threshold = REAL(0.5);
    const Matrix matrix = make_layout_matrix(perturb, 0, variant);
    const size_t nnz = matrix.values.size();
    // Use a dense coordinate lookup as an independent map oracle, rather than
    // duplicating the production builder's sorted-column merge.
    const unsigned missing = std::numeric_limits<unsigned>::max();
    std::vector<unsigned> positions(size_t(matrix.n) * matrix.n, missing);
    for (unsigned c = 0; c < matrix.n; ++c)
        for (unsigned p = matrix.column[c]; p < matrix.column[c + 1]; ++p)
            positions[size_t(c) * matrix.n + matrix.row[p]] = p;
    std::vector<uint64_t> bases(matrix.n + 1);
    std::vector<unsigned> destinations, pivots(matrix.csr_column.size(), missing);
    for (unsigned c = 0; c < matrix.n; ++c)
    {
        bases[c] = destinations.size();
        for (unsigned q = matrix.csr_diagonal[c] + 1; q < matrix.csr_row[c + 1]; ++q)
        {
            const unsigned target = matrix.csr_column[q];
            pivots[q] = positions[size_t(target) * matrix.n + c];
            if (pivots[q] == missing)
                throw std::runtime_error("mapped fixture missing U entry");
            for (unsigned p = matrix.diagonal[c] + 1; p < matrix.column[c + 1]; ++p)
            {
                const unsigned destination = positions[size_t(target) * matrix.n + matrix.row[p]];
                if (destination == missing)
                    throw std::runtime_error("mapped fixture is not closed");
                destinations.push_back(destination);
            }
        }
        std::vector<unsigned> unique(destinations.begin() + bases[c], destinations.end());
        std::sort(unique.begin(), unique.end());
        if (std::adjacent_find(unique.begin(), unique.end()) != unique.end())
            throw std::runtime_error("one mapped source has duplicate update destinations");
    }
    bases[matrix.n] = destinations.size();
    std::vector<REAL> input, expected;
    for (unsigned copy = 0; copy < copies; ++copy)
    {
        const Matrix distinct = make_layout_matrix(perturb, copy, variant);
        input.insert(input.end(), distinct.values.begin(), distinct.values.end());
        const auto result = reference_update(distinct, perturb, threshold);
        expected.insert(expected.end(), result.begin(), result.end());
    }
    DeviceBuffer<unsigned> column(matrix.column), row(matrix.row), diagonal(matrix.diagonal);
    DeviceBuffer<unsigned> csr_row(matrix.csr_row), csr_column(matrix.csr_column), csr_diagonal(matrix.csr_diagonal);
    DeviceBuffer<uint64_t> map_bases(bases);
    DeviceBuffer<unsigned> map_destinations(destinations), map_pivots(pivots);
    // Nonzero levelHead plus a second chunk exercises both source-list offsets.
    DeviceBuffer<int> sources(std::vector<int>{0, 2, 5, 8, 11});
    std::vector<uint64_t> target_prefix{0, 7};
    for (unsigned c : {2u, 5u, 8u, 11u})
        target_prefix.push_back(
            target_prefix.back() +
            (matrix.column[c + 1] > matrix.diagonal[c] + 1 ? matrix.csr_row[c + 1] - matrix.csr_diagonal[c] - 1 : 0));
    DeviceBuffer<uint64_t> prefix(target_prefix);
    DeviceBuffer<REAL> values(input), interleaved(soa ? nnz * pitch : 1);
    Stream stream;
    for (unsigned repeat = 0; repeat < 2; ++repeat)
    {
        const REAL guard = repeat ? std::numeric_limits<REAL>::quiet_NaN() : REAL(12345.5);
        const std::vector<REAL> guards(soa ? nnz * pitch : 1, guard);
        check_gpu(cudaMemcpy(interleaved.data(), guards.data(), guards.size() * sizeof(REAL), cudaMemcpyHostToDevice),
            "initialize guards");
        check_gpu(cudaMemcpy(values.data(), input.data(), input.size() * sizeof(REAL), cudaMemcpyHostToDevice),
            "restore oracle inputs");
        const unsigned tiles = unsigned((nnz / 32 + (nnz % 32 != 0)) * (copies / 32 + (copies % 32 != 0)));
        if (soa)
            transpose_task_values<<<tiles, dim3(32, 8), 0, stream.value>>>(values.data(), interleaved.data(), copies,
                unsigned(nnz), unsigned(nnz), pitch);
        if (normalized)
        {
            if (perturb)
                mapped_normalize_level_soa<true><<<dim3(4, tiles_y), 256, 0, stream.value>>>(interleaved.data(), copies,
                    column.data(), diagonal.data(), sources.data(), 1, bits, threshold, pitch);
            else
                mapped_normalize_level_soa<false><<<dim3(4, tiles_y), 256, 0, stream.value>>>(interleaved.data(),
                    copies, column.data(), diagonal.data(), sources.data(), 1, bits, 0, pitch);
        }
        const unsigned targets = unsigned(target_prefix.back() - 7);
        if (soa)
        {
            if (normalized)
                mapped_level_update_soa<false, true><<<dim3(targets, tiles_y), 256, 0, stream.value>>>(
                    interleaved.data(), copies, column.data(), diagonal.data(), csr_diagonal.data(), sources.data(),
                    prefix.data(), 1, 5, map_bases.data(), map_destinations.data(), map_pivots.data(), 0, pitch, bits);
            else if (perturb)
                mapped_level_update_soa<true><<<dim3(targets, tiles_y), 256, 0, stream.value>>>(interleaved.data(),
                    copies, column.data(), diagonal.data(), csr_diagonal.data(), sources.data(), prefix.data(), 1, 5,
                    map_bases.data(), map_destinations.data(), map_pivots.data(), threshold, pitch, bits);
            else
                mapped_level_update_soa<false><<<dim3(targets, tiles_y), 256, 0, stream.value>>>(interleaved.data(),
                    copies, column.data(), diagonal.data(), csr_diagonal.data(), sources.data(), prefix.data(), 1, 5,
                    map_bases.data(), map_destinations.data(), map_pivots.data(), 0, pitch, bits);
        }
        else
        {
            if (perturb)
                mapped_level_update<true><<<dim3(targets, copies), 256, 0, stream.value>>>(values.data(), nnz,
                    column.data(), diagonal.data(), csr_diagonal.data(), sources.data(), prefix.data(), 1, 5,
                    map_bases.data(), map_destinations.data(), map_pivots.data(), threshold);
            else
                mapped_level_update<false><<<dim3(targets, copies), 256, 0, stream.value>>>(values.data(), nnz,
                    column.data(), diagonal.data(), csr_diagonal.data(), sources.data(), prefix.data(), 1, 5,
                    map_bases.data(), map_destinations.data(), map_pivots.data(), 0);
        }
        if (soa)
            transpose_task_values<<<tiles, dim3(32, 8), 0, stream.value>>>(interleaved.data(), values.data(),
                unsigned(nnz), copies, pitch, unsigned(nnz));
        if (!normalized)
        {
            if (perturb)
                RL_onecol_factorizeCurrentCol<true, false, true>
                    <<<dim3(4, copies), 256, 0, stream.value>>>(column.data(), row.data(), values.data(),
                        diagonal.data(), 0, nullptr, 0, matrix.n, nnz, 0, threshold, sources.data(), 1);
            else
                RL_onecol_factorizeCurrentCol<false, false, true>
                    <<<dim3(4, copies), 256, 0, stream.value>>>(column.data(), row.data(), values.data(),
                        diagonal.data(), 0, nullptr, 0, matrix.n, nnz, 0, 0, sources.data(), 1);
        }
        check_gpu(cudaGetLastError(), "launch layout oracle kernels");
        check_gpu(cudaStreamSynchronize(stream.value), "wait layout oracle");
        compare(values.read(), expected, "layout factors and distinct values");
        if (soa)
        {
            const auto actual = interleaved.read();
            for (size_t row = 0; row < nnz; ++row)
                for (unsigned copy = copies; copy < pitch; ++copy)
                    if (repeat ? !std::isnan(actual[row * pitch + copy]) : actual[row * pitch + copy] != guard)
                        throw std::runtime_error("padding guard overwritten");
        }
    }
}

void test_small_level(bool graph, unsigned copies, bool perturb)
{
    // First level has four sources: zero targets, two targets, empty L, one
    // target. The repeated prefixes are {0,0,2,2,3}; sources 1/3 overlap in col5.
    unsigned columns[]{0, 2, 5, 6, 8, 10, 15};
    unsigned rows[]{0, 5, 1, 4, 5, 2, 3, 5, 1, 4, 1, 2, 3, 4, 5};
    double values[]{4, 2, 8, 1, 2, 16, 32, 2, 2, 20, 1, 4, 2, 1, 30};
    if (perturb)
        values[5] = 0;
    std::ostringstream out, err;
    Symbolic_Matrix matrix(6, out, err);
    matrix.fill_in(rows, columns);
    matrix.csr();
    matrix.predictLU(rows, columns, values);
    matrix.leveling();
    if (matrix.level_ptr[1] != 4)
        throw std::runtime_error("small-level fixture width changed");
    GpuMatrix control(matrix, 1, perturb, UpdateMode::Scan);
    GpuTask reference(control, copies);
    reference.enqueue();
    reference.wait();
    reference.check();
    GpuMatrix candidate(matrix, 1, perturb, UpdateMode::Mapped);
    verify_template_bytes(candidate, matrix, UpdateMode::Mapped);
    GpuTask task(candidate, copies);
    if (graph)
        task.prepare_graph();
    for (unsigned loop = 0; loop < 3; ++loop)
    {
        task.enqueue();
        task.wait();
        task.check();
        for (unsigned copy = 0; copy < copies; ++copy)
        {
            compare(std::vector<REAL>(task.factors(copy), task.factors(copy) + matrix.nnz),
                std::vector<REAL>(reference.factors(copy), reference.factors(copy) + matrix.nnz),
                "two-stage unequal targets and empty L");
            if (!perturb && matrix.factor_residual(task.factors(copy)) > (sizeof(REAL) == 4 ? 1e-5 : 1e-12))
                throw std::runtime_error("two-stage full factor residual");
        }
    }
}

// Test-only scalar geometry control; production uses MMAC or its wave fallback.
__global__ void mapped_dense_scalar(REAL* values,
    unsigned copies,
    unsigned stride,
    unsigned bits,
    const DenseUpdateTile* tiles,
    const unsigned* indices,
    unsigned begin)
{
    const unsigned lanes = 1u << bits, copy = blockIdx.y * lanes + (threadIdx.x & (lanes - 1));
    if (copy < copies)
        scalar_dense_outputs(values, tiles[begin + blockIdx.x], indices, copy, stride, threadIdx.x >> bits,
            blockDim.x >> bits);
}

void test_dense_updates_case(unsigned length, bool full_mantissa)
{
    // Independent coordinate fixture: no production planner or factorization
    // oracle. Four views cover a partial 17x19 rectangle; K is varied separately.
    constexpr unsigned nr = 17, nc = 19, stride = 8;
    constexpr unsigned dst_base = 7, lower_base = dst_base + nr * nc + 3;
    const unsigned pivot_base = lower_base + nr * length + 3;
    const unsigned positions = pivot_base + nc * length + 7;
#if defined(GLU_USE_FP32)
    // Test-side operand sensitivity proxy, not an emulation of a backend's
    // accumulation order: round binary32 operands to ten fraction bits, RNE.
    const auto tf32_rne = [](REAL value)
    {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        if ((bits & 0x7f800000u) != 0x7f800000u)
        {
            bits += 0x00000fffu + ((bits >> 13) & 1u);
            bits &= 0xffffe000u;
        }
        std::memcpy(&value, &bits, sizeof(bits));
        return value;
    };
#endif
    const auto hole = [](unsigned r, unsigned c)
    { return (r == 3 && c == 2) || (r == 5 && c == 17) || (r == 16 && c == 18); };
    // Prefix guards also exercise nonzero index-array offsets.
    std::vector<unsigned> indices(3, UINT32_MAX);
    const unsigned dst_begin = unsigned(indices.size());
    for (unsigned c = 0; c < nc; ++c)
        for (unsigned r = 0; r < nr; ++r)
            indices.push_back(hole(r, c) ? UINT32_MAX : dst_base + c * nr + r);
    const unsigned lower_begin = unsigned(indices.size());
    for (unsigned q = 0; q < length; ++q)
        for (unsigned r = 0; r < nr; ++r)
            indices.push_back(lower_base + q * nr + r);
    const unsigned pivot_begin = unsigned(indices.size());
    for (unsigned q = 0; q < length; ++q)
        for (unsigned c = 0; c < nc; ++c)
            indices.push_back(pivot_base + q * nc + c);
    std::vector<DenseUpdateTile> tiles;
    for (unsigned c = 0; c < nc; c += 16)
        for (unsigned r = 0; r < nr; r += 16)
            tiles.push_back({dst_begin, lower_begin, pivot_begin, nr, nc, length, r, c});
    if (tiles.size() != 4)
        throw std::logic_error("dense fixture tile count");
    DeviceBuffer<unsigned> device_indices(indices);
    DeviceBuffer<DenseUpdateTile> device_tiles(tiles);
    Stream stream;
    for (unsigned copies : {1u, 3u, 5u})
        for (bool nonfinite : {false, true})
        {
            std::vector<REAL> input(size_t(positions) * stride);
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = REAL(-32) - REAL(i % 31) / REAL(32);
            for (unsigned copy = 0; copy < copies; ++copy)
            {
                for (unsigned c = 0; c < nc; ++c)
                    for (unsigned r = 0; r < nr; ++r)
                        if (!hole(r, c))
                            input[size_t(dst_base + c * nr + r) * stride + copy] =
                                full_mantissa
                                    ? REAL(2.0 + double(r + 1) / 997.0 + double(c + 1) / 1009.0 +
                                           double(copy + 1) / 1013.0)
                                    : REAL(2) + REAL(r) / REAL(16) + REAL(c) / REAL(32) + REAL(copy) / REAL(8);
                for (unsigned q = 0; q < length; ++q)
                    for (unsigned r = 0; r < nr; ++r)
                    {
                        const int numerator = int((3 * r + 5 * q + 7 * copy) % 17) - 8;
                        REAL factor = full_mantissa
                                          ? REAL(double(numerator) / 29.0 + double(r + 3 * q + copy + 1) / 8191.0)
                                          : REAL(numerator) / REAL(64);
                        if (q == 1)
                            factor = full_mantissa ? REAL(1.0 + double(r + copy + 1) / 1009.0)
                                                   : REAL(1) + REAL(r + copy) / REAL(64); // Always a real update.
                        if (nonfinite && q == 0)
                            factor = REAL(0);
                        input[size_t(lower_base + q * nr + r) * stride + copy] = factor;
                    }
                for (unsigned q = 0; q < length; ++q)
                    for (unsigned c = 0; c < nc; ++c)
                    {
                        REAL pivot = full_mantissa ? REAL(1.0 + double(2 * c + q + 3 * copy + 1) / 997.0)
                                                   : REAL(1) + REAL(2 * c + q + 3 * copy) / REAL(64);
                        if (nonfinite && q == 0 && copy % 2 == 0)
                        {
                            if (c == 1)
                                pivot = std::numeric_limits<REAL>::infinity();
                            if (c == 17)
                                pivot = std::numeric_limits<REAL>::quiet_NaN();
                        }
                        input[size_t(pivot_base + q * nc + c) * stride + copy] = pivot;
                    }
            }
            std::vector<long double> expected(input.size(), 0);
            std::vector<unsigned char> writable(input.size(), 0);
#if defined(GLU_USE_FP32)
            unsigned tf32_distinguished = 0;
#endif
            for (unsigned copy = 0; copy < copies; ++copy)
                for (unsigned c = 0; c < nc; ++c)
                    for (unsigned r = 0; r < nr; ++r)
                    {
                        if (hole(r, c))
                            continue;
                        const size_t dst = size_t(dst_base + c * nr + r) * stride + copy;
                        long double acc = input[dst];
#if defined(GLU_USE_FP32)
                        long double tf32_acc = acc;
#endif
                        for (unsigned q = 0; q < length; ++q)
                        {
                            const REAL factor = input[size_t(lower_base + q * nr + r) * stride + copy];
                            // In particular, never evaluate zero times Inf or NaN.
                            if (factor != REAL(0))
                                acc -= static_cast<long double>(factor) *
                                       static_cast<long double>(input[size_t(pivot_base + q * nc + c) * stride + copy]);
#if defined(GLU_USE_FP32)
                            if (full_mantissa && !nonfinite)
                            {
                                const REAL rounded = tf32_rne(factor);
                                if (rounded != REAL(0))
                                    tf32_acc -= static_cast<long double>(rounded) *
                                                static_cast<long double>(
                                                    tf32_rne(input[size_t(pivot_base + q * nc + c) * stride + copy]));
                            }
#endif
                        }
                        if (!std::isfinite(acc))
                            throw std::logic_error("dense oracle must remain finite");
                        expected[dst] = acc;
                        writable[dst] = 1;
#if defined(GLU_USE_FP32)
                        const long double tolerance = 64 *
                                                      static_cast<long double>(std::numeric_limits<REAL>::epsilon()) *
                                                      std::max(1.0L, std::fabs(acc));
                        if (full_mantissa && !nonfinite && std::fabs(tf32_acc - acc) > 8 * tolerance)
                            ++tf32_distinguished;
#endif
                    }
#if defined(GLU_USE_FP32)
            if (full_mantissa && !nonfinite && tf32_distinguished < 8)
                throw std::logic_error(
                    "dense full-mantissa fixture lacks TF32 operand sensitivity, K=" + std::to_string(length));
#endif
            unsigned bits = 0;
            while ((1u << bits) < copies)
                ++bits;
            for (bool tiled : {false, true})
            {
                DeviceBuffer<REAL> values(input);
                if (tiled)
                    panel_tiled_update<<<dim3(4, (copies + 3) / 4), 256, 0, stream.value>>>(values.data(), copies,
                        stride, device_tiles.data(), device_indices.data(), 0);
                else
                    mapped_dense_scalar<<<dim3(4, (copies + (1u << bits) - 1) / (1u << bits)), 256, 0, stream.value>>>(
                        values.data(), copies, stride, bits, device_tiles.data(), device_indices.data(), 0);
                check_gpu(cudaGetLastError(), "launch independent dense oracle");
                check_gpu(cudaStreamSynchronize(stream.value), "wait independent dense oracle");
                const auto actual = values.read();
                for (size_t i = 0; i < actual.size(); ++i)
                {
                    bool valid = true;
                    if (writable[i])
                    {
                        const long double tolerance = 64 *
                                                      static_cast<long double>(std::numeric_limits<REAL>::epsilon()) *
                                                      std::max(1.0L, std::fabs(expected[i]));
                        valid = std::isfinite(actual[i]) &&
                                std::fabs(static_cast<long double>(actual[i]) - expected[i]) <= tolerance;
                    }
                    else
                    {
                        // Bitwise preservation handles NaN operands as well as all
                        // L/U, physical D holes, padding copies and outside guards.
                        const auto* before = reinterpret_cast<const unsigned char*>(&input[i]);
                        const auto* after = reinterpret_cast<const unsigned char*>(&actual[i]);
                        for (size_t b = 0; b < sizeof(REAL); ++b)
                            if (before[b] != after[b])
                                valid = false;
                    }
                    if (!valid)
                    {
                        std::ostringstream error;
                        error << "dense " << (tiled ? "tiled" : "scalar") << " copies=" << copies << " K=" << length
                              << " full_mantissa=" << full_mantissa << " nonfinite=" << nonfinite << " position=" << i
                              << (writable[i] ? " output mismatch" : " unexpected write");
                        if (writable[i])
                            error << std::setprecision(std::numeric_limits<long double>::max_digits10)
                                  << " actual=" << static_cast<long double>(actual[i]) << " expected=" << expected[i];
                        throw std::runtime_error(error.str());
                    }
                }
            }
        }
}

void test_dense_updates()
{
    test_dense_updates_case(5, false); // Preserve the original binary-grid control.
    for (unsigned length : {3u, 4u, 5u, 9u})
        test_dense_updates_case(length, true);
    std::cout << "PASS dense/independent-coordinate-oracle-full-mantissa-holes-tail-and-nonfinite\n";
}

void test_panel_boundaries()
{
    // Dense, well-conditioned known L/U: >2 windows, a partial tail, and
    // trailing D/L/U positions consumed only after an earlier panel drains.
    constexpr unsigned n = 67;
    std::vector<unsigned> columns{0}, rows;
    std::vector<double> input, expected;
    auto l = [](unsigned r, unsigned c) { return r == c ? 1.0 : r > c ? double((r + 2 * c) % 7 + 1) / 128.0 : 0.0; };
    auto u = [](unsigned r, unsigned c) {
        return r == c ? 8.0 + double(r % 3) : r < c ? double((2 * r + c) % 5 - 0u) / 32.0 : 0.0;
    };
    for (unsigned c = 0; c < n; ++c)
    {
        for (unsigned r = 0; r < n; ++r)
        {
            double value = 0;
            for (unsigned k = 0; k < n; ++k)
                value += l(r, k) * u(k, c);
            rows.push_back(r);
            input.push_back(value);
            expected.push_back(r > c ? l(r, c) : u(r, c));
        }
        columns.push_back(unsigned(rows.size()));
    }
    std::ostringstream out, err;
    Symbolic_Matrix matrix(n, out, err);
    matrix.fill_in(rows.data(), columns.data());
    matrix.csr();
    matrix.predictLU(rows.data(), columns.data(), input.data());
    matrix.leveling();
    if (matrix.num_lev != int(n))
        throw std::runtime_error("panel chain fixture changed");
    for (UpdateMode mode : {UpdateMode::Panel, UpdateMode::PanelTiled})
        for (unsigned copies : {1u, 3u, 33u})
            for (bool graph : {false, true})
                for (bool perturb : {false, true})
                {
                    GpuMatrix gpu(matrix, 1, perturb, mode);
                    GpuTask task(gpu, copies);
                    if (graph)
                        task.prepare_graph();
                    for (unsigned repeat = 0; repeat < 2; ++repeat)
                    {
                        task.enqueue();
                        task.wait();
                        task.check();
                        for (unsigned copy = 0; copy < copies; ++copy)
                        {
                            compare(std::vector<REAL>(task.factors(copy), task.factors(copy) + matrix.nnz),
                                std::vector<REAL>(expected.begin(), expected.end()), "known LU across panels");
                            if (matrix.factor_residual(task.factors(copy)) > (sizeof(REAL) == 8 ? 1e-12 : 1e-5))
                                throw std::runtime_error("panel full residual");
                        }
                    }
                }
}

void test_fused_ordinary_groups()
{
    // Force the ordinary-record kernel without a huge packing-limit fixture.
    // Known L/U generate A by dense multiplication, independently of the plan.
    constexpr unsigned n = 40, copies = 3;
    std::vector<unsigned> columns{0}, rows;
    std::vector<double> initial;
    std::vector<REAL> input, expected;
    for (unsigned copy = 0; copy < copies; ++copy)
    {
        const auto l = [copy](unsigned r, unsigned c) {
            return r == c ? 1.0 : r > c ? double((r + 2 * c + copy) % 7) / 128.0 : 0.0;
        };
        const auto u = [copy](unsigned r, unsigned c)
        {
            return r == c  ? 8.0 + double(r % 3) + double(copy) / 4
                   : r < c ? double(int((2 * r + c + copy) % 5) - 2) / 32.0
                           : 0.0;
        };
        for (unsigned c = 0; c < n; ++c)
        {
            for (unsigned r = 0; r < n; ++r)
            {
                double value = 0;
                for (unsigned k = 0; k < n; ++k)
                    value += l(r, k) * u(k, c);
                input.push_back(REAL(value));
                expected.push_back(REAL(r > c ? l(r, c) : u(r, c)));
                if (copy == 0)
                {
                    rows.push_back(r);
                    initial.push_back(value);
                }
            }
            if (copy == 0)
                columns.push_back(unsigned(rows.size()));
        }
    }
    std::ostringstream out, err;
    Symbolic_Matrix matrix(n, out, err);
    matrix.fill_in(rows.data(), columns.data());
    matrix.csr();
    matrix.predictLU(rows.data(), columns.data(), initial.data());
    matrix.leveling();
    if (matrix.num_lev != int(n) || matrix.sym_c_ptr != columns || matrix.sym_r_idx != rows)
        throw std::runtime_error("ordinary Fused fixture lost its dense chain layout");
    DeviceBuffer<unsigned> device_diagonals(matrix.l_col_ptr);
    DeviceBuffer<int> device_order(matrix.level_idx);
    Stream stream;
    for (bool inline_singletons : {false, true})
    {
        auto plan = make_panel_update_plan(matrix, 32, false, inline_singletons);
        if (plan.immediate[0].end - plan.immediate[0].begin <= 256 || plan.deferred[31].begin == plan.deferred[31].end)
            throw std::runtime_error("ordinary Fused fixture lost immediate or deferred updates");
        const auto strip_plan = organize_fused_updates(plan, inline_singletons);
        std::vector<FusedStep> steps;
        std::vector<FusedNormalization> normalizations;
        for (unsigned lev = 0; lev < n; ++lev)
        {
            const unsigned begin = unsigned(normalizations.size());
            for (int p = matrix.level_ptr[lev]; p < matrix.level_ptr[lev + 1]; ++p)
            {
                const unsigned column = unsigned(matrix.level_idx[p]), diagonal = matrix.l_col_ptr[column];
                for (unsigned pos = diagonal + 1; pos < matrix.sym_c_ptr[column + 1]; ++pos)
                    normalizations.push_back({pos, diagonal});
            }
            steps.push_back(
                {{unsigned(matrix.level_ptr[lev]), unsigned(matrix.level_ptr[lev + 1])}, plan.immediate[lev],
                    plan.deferred[lev], strip_plan.ranges[lev], {begin, unsigned(normalizations.size())}});
        }
        DeviceBuffer<FusedStep> device_steps(steps);
        DeviceBuffer<UpdateGroup> device_groups(plan.groups);
        DeviceBuffer<NormalizedUpdatePair> device_pairs(plan.pairs);
        DeviceBuffer<FusedNormalization> device_normalizations(normalizations);
        DeviceBuffer<DeferredUpdateStrip> device_strips(strip_plan.strips);
        DeviceBuffer<unsigned> device_destinations(strip_plan.destinations);
        DeviceBuffer<REAL> device_values(input);
        fused_panel_factorize<false, false><<<copies, inline_singletons ? 1024 : 256, 0, stream.value>>>(
            device_values.data(), matrix.nnz, device_diagonals.data(), device_order.data(), device_steps.data(), n,
            device_groups.data(), nullptr, device_pairs.data(), device_strips.data(), device_destinations.data(),
            device_normalizations.data(), inline_singletons, REAL(0));
        check_gpu(cudaGetLastError(), "launch ordinary Fused regression");
        check_gpu(cudaStreamSynchronize(stream.value), "wait ordinary Fused regression");
        compare(device_values.read(), expected, "ordinary Fused independent known LU");
    }
    std::cout << "PASS fused/ordinary-groups-immediate-deferred-inline-and-distinct-copies\n";
}

void test_fused_strip_operands()
{
    std::vector<REAL> values(3, REAL(1234)), expected;
    std::vector<unsigned> destinations;
    std::vector<NormalizedUpdatePair> pairs;
    std::vector<DeferredUpdateStrip> strips;
    for (const auto shape : std::vector<std::pair<unsigned, unsigned>>{{1, 2}, {17, 3}, {31, 4}, {32, 7}})
    {
        const unsigned count = shape.first, length = shape.second, first = unsigned(destinations.size()),
                       base = unsigned(pairs.size());
        for (unsigned lane = 0; lane < count; ++lane)
        {
            destinations.push_back(unsigned(values.size()));
            values.push_back(REAL(2) + REAL(lane) / 64);
            expected.push_back(values.back());
        }
        for (unsigned q = 0; q < length; ++q)
            for (unsigned lane = 0; lane < count; ++lane)
            {
                const REAL lower = q % 3 == 0 ? (lane % 2 ? -REAL(0) : REAL(0)) : REAL(1 + lane % 5) / 64;
                const REAL pivot = q % 3 == 0 ? (lane % 2 ? std::numeric_limits<REAL>::infinity()
                                                          : std::numeric_limits<REAL>::quiet_NaN())
                                              : REAL(1 + q + lane % 7) / 32;
                const unsigned at = unsigned(values.size());
                values.push_back(lower);
                values.push_back(pivot);
                pairs.push_back({at, at + 1});
                if (lower != REAL(0))
                    expected[first + lane] -= lower * pivot;
            }
        strips.push_back({first, base, count, length});
    }
    const std::vector<FusedStep> steps{{{0, 0}, {0, 0}, {0, 0}, {0, unsigned(strips.size())}, {0, 0}}};
    DeviceBuffer<FusedStep> device_steps(steps);
    DeviceBuffer<NormalizedUpdatePair> device_pairs(pairs);
    DeviceBuffer<DeferredUpdateStrip> device_strips(strips);
    DeviceBuffer<unsigned> device_destinations(destinations);
    Stream stream;
    for (unsigned threads : {256u, 1024u})
    {
        DeviceBuffer<REAL> data(values);
        fused_panel_factorize<false, true><<<1, threads, 0, stream.value>>>(data.data(), unsigned(values.size()),
            nullptr, nullptr, device_steps.data(), 1, nullptr, nullptr, device_pairs.data(), device_strips.data(),
            device_destinations.data(), nullptr, false, REAL(0));
        check_gpu(cudaGetLastError(), "launch Fused strip operand regression");
        check_gpu(cudaStreamSynchronize(stream.value), "wait Fused strip operand regression");
        const auto result = data.read();
        std::vector<REAL> actual;
        for (unsigned p : destinations)
            actual.push_back(result[p]);
        compare(actual, expected, "Fused strips: odd lengths, masked lanes and zero times nonfinite");
    }
    std::cout << "PASS fused-strip-operands-and-zero-factor-skip\n";
}

template <typename Exception, typename Operation>
void expect_task_rejection(Operation operation, const char* diagnostic)
{
    bool rejected = false;
    try
    {
        operation();
    }
    catch (const Exception&)
    {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error(diagnostic);
}

void test_submission_failure_cleanup()
{
    constexpr unsigned copies = 2;
    unsigned columns[]{0, 2, 4}, rows[]{0, 1, 0, 1};
    double values[]{4, 2, 1, 3}; // Closed 2x2 pattern with nonzero known LU pivots.
    std::ostringstream out, err;
    Symbolic_Matrix matrix(2, out, err);
    matrix.fill_in(rows, columns);
    matrix.csr();
    matrix.predictLU(rows, columns, values);
    matrix.leveling();
    GpuMatrix gpu(matrix, 1, false, UpdateMode::Scan);
    Buffer<REAL, true> input;
    const size_t elements = size_t(matrix.nnz) * copies;
    input.allocate(elements);
    for (unsigned copy = 0; copy < copies; ++copy)
        std::copy(matrix.val.begin(), matrix.val.end(), input.ptr + size_t(copy) * matrix.nnz);

    for (bool fail_upload : {true, false})
        for (bool override_sync : {false, true})
        {
            GpuTask task(gpu, copies);
            async_fault::State state{fail_upload, override_sync};
            async_fault::Scope fault(state);
            if (!fail_upload)
            {
                task.upload_input(input.ptr, elements);
                if (state.sync_calls != 0)
                    throw std::runtime_error("successful fresh upload unexpectedly synchronized");
            }
            std::string diagnostic;
            try
            {
                if (fail_upload)
                    task.upload_input(input.ptr, elements);
                else
                    task.enqueue();
            }
            catch (const std::runtime_error& error)
            {
                diagnostic = error.what();
            }
            const char* operation = fail_upload ? "upload matrix input" : "download matrix factors";
            const std::string expected = std::string(operation) + ": " + cudaGetErrorString(async_fault::copy_error);
            if (diagnostic != expected)
                throw std::runtime_error("submission failure lost original error: " + diagnostic);
            // Assert before task destruction can drain anything. The pinned AoS
            // source remains alive, including while enqueue drains the batch upload.
            if (state.h2d_calls != 1 || state.h2d_submitted != 1 || state.d2h_calls != (fail_upload ? 0u : 1u) ||
                !state.same_stream || state.sync_calls != 1 || state.sync_successes != 1)
                throw std::runtime_error("submission failure did not drain the batch upload on its stream");
            expect_task_rejection<std::logic_error>([&] { task.enqueue(); }, "failed task accepted enqueue");
            expect_task_rejection<std::logic_error>([&] { task.upload_input(); },
                "failed task accepted default upload");
            expect_task_rejection<std::logic_error>([&] { task.upload_input(input.ptr, elements); },
                "failed task accepted fresh upload");
            expect_task_rejection<std::logic_error>([&] { task.prepare_graph(); },
                "failed task accepted graph preparation");
            expect_task_rejection<std::logic_error>([&] { task.factors(); }, "failed task allowed result access");
        }
    std::cout << "PASS host/submission-failure-drain-and-original-error\n";
}

void test_task_set_submission_failure_cleanup()
{
    unsigned columns[]{0, 2, 4}, rows[]{0, 1, 0, 1};
    double values[]{4, 2, 1, 3};
    std::ostringstream out, err;
    Symbolic_Matrix matrix(2, out, err);
    matrix.fill_in(rows, columns);
    matrix.csr();
    matrix.predictLU(rows, columns, values);
    matrix.leveling();
    // A distinct compact order and omitted upper slot exercise both upload paths.
    const std::vector<unsigned> positions{3, 1, 0};
    GpuMatrix gpu(matrix, positions, 1, false, UpdateMode::Fused);
    Buffer<REAL, true> full, compact;
    constexpr size_t count = 2;
    full.allocate(matrix.nnz, count);
    compact.allocate(positions.size(), count);
    for (size_t task = 0; task < count; ++task)
    {
        std::copy(matrix.val.begin(), matrix.val.end(), full.ptr + task * matrix.nnz);
        for (size_t p = 0; p < positions.size(); ++p)
            compact.ptr[task * positions.size() + p] = matrix.val[positions[p]];
    }

    for (bool use_compact : {false, true})
        for (bool fail_upload : {true, false})
            for (bool override_sync : {false, true})
            {
                GpuTaskSet tasks({&gpu}, {0, 0});
                async_fault::State state{fail_upload, override_sync};
                async_fault::Scope fault(state);
                const auto upload = [&]
                {
                    if (use_compact)
                        tasks.upload_compact_input(compact.ptr, count * positions.size());
                    else
                        tasks.upload_input(full.ptr, count * matrix.nnz);
                };
                if (!fail_upload)
                {
                    upload();
                    if (state.sync_calls != 0)
                        throw std::runtime_error("successful task set upload unexpectedly synchronized");
                }
                std::string diagnostic;
                try
                {
                    if (fail_upload)
                        upload();
                    else
                        tasks.enqueue();
                }
                catch (const std::runtime_error& error)
                {
                    diagnostic = error.what();
                }
                const char* operation = fail_upload ? "upload task set input" : "download task set factors";
                const std::string expected =
                    std::string(operation) + ": " + cudaGetErrorString(async_fault::copy_error);
                if (diagnostic != expected)
                    throw std::runtime_error("task set submission failure lost original error: " + diagnostic);
                // Check the drain before wait/destruction can mask a missing cleanup.
                // Both pinned source buffers remain alive through all submitted work.
                if (state.h2d_calls != 1 || state.h2d_submitted != 1 || state.d2h_calls != (fail_upload ? 0u : 1u) ||
                    !state.same_stream || state.sync_calls != 1 || state.sync_successes != 1)
                    throw std::runtime_error("task set submission failure did not drain its stream");
                expect_task_rejection<std::logic_error>([&] { tasks.enqueue(); }, "failed task set accepted enqueue");
                expect_task_rejection<std::logic_error>([&] { tasks.upload_input(); },
                    "failed task set accepted default upload");
                expect_task_rejection<std::logic_error>([&] { tasks.upload_input(full.ptr, count * matrix.nnz); },
                    "failed task set accepted full upload");
                expect_task_rejection<std::logic_error>([&]
                    { tasks.upload_compact_input(compact.ptr, count * positions.size()); },
                    "failed task set accepted compact upload");
                expect_task_rejection<std::logic_error>([&] { tasks.factors(0); },
                    "failed task set allowed result access");
                expect_task_rejection<std::logic_error>([&] { tasks.check(); },
                    "failed task set allowed factor checks");
                expect_task_rejection<std::runtime_error>([&] { tasks.wait(); }, "failed task set accepted wait");
            }
    std::cout << "PASS host/task-set-full-compact-failure-drain-and-original-error\n";
}

void test_fresh_inputs()
{
    // A small closed pattern keeps every round in the same factor-slot layout.
    // L/U are independent known factors; both round and copy affect their values.
    constexpr unsigned n = 9;
    const auto make_values = [](unsigned round, unsigned copy)
    {
        const auto l = [=](unsigned r, unsigned c)
        {
            return r == c  ? 1.0
                   : r > c ? double((r + 2 * c + 3 * round + 5 * copy) % 7 + 1) / 128.0 + double(round + 1) / 512.0 +
                                 double(copy + 1) / 8192.0
                           : 0.0;
        };
        const auto u = [=](unsigned r, unsigned c)
        {
            return r == c  ? 8.0 + double(r % 3) + double(round) / 4.0 + double(copy) / 64.0
                   : r < c ? double((2 * r + c + round + 2 * copy) % 5) / 32.0 + double(round + 1) / 128.0 +
                                 double(copy + 1) / 2048.0
                           : 0.0;
        };
        std::vector<double> input;
        std::vector<REAL> expected;
        for (unsigned c = 0; c < n; ++c)
            for (unsigned r = 0; r < n; ++r)
            {
                double value = 0;
                for (unsigned k = 0; k < n; ++k)
                    value += l(r, k) * u(k, c);
                input.push_back(value);
                expected.push_back(REAL(r > c ? l(r, c) : u(r, c)));
            }
        return std::make_pair(std::move(input), std::move(expected));
    };
    auto initial = make_values(0, 0);
    std::vector<unsigned> columns{0}, rows;
    for (unsigned c = 0; c < n; ++c)
    {
        for (unsigned r = 0; r < n; ++r)
            rows.push_back(r);
        columns.push_back(unsigned(rows.size()));
    }
    std::ostringstream out, err;
    Symbolic_Matrix matrix(n, out, err);
    matrix.fill_in(rows.data(), columns.data());
    matrix.csr();
    matrix.predictLU(rows.data(), columns.data(), initial.first.data());
    matrix.leveling();
    if (matrix.sym_c_ptr != columns || matrix.sym_r_idx != rows)
        throw std::runtime_error("fresh input fixture changed factor-slot layout");

    for (UpdateMode mode : {UpdateMode::Scan, UpdateMode::Mapped, UpdateMode::Panel, UpdateMode::PanelTiled})
        for (SnapshotPolicy snapshot :
            {SnapshotPolicy::Auto, SnapshotPolicy::PerCopy, SnapshotPolicy::Shared, SnapshotPolicy::Device})
            for (unsigned copies : {1u, 3u, 33u})
                for (bool graph : {false, true})
                {
                    try
                    {
                        GpuMatrix gpu(matrix, 1, false, mode, snapshot);
                        GpuTask task(gpu, copies);
                        if (graph)
                            task.prepare_graph(); // Once, before every subsequent fresh input.
                        const size_t elements = size_t(matrix.nnz) * copies;
                        std::vector<REAL> default_input, default_factors;
                        for (unsigned copy = 0; copy < copies; ++copy)
                        {
                            default_input.insert(default_input.end(), matrix.val.begin(), matrix.val.end());
                            default_factors.insert(default_factors.end(), initial.second.begin(), initial.second.end());
                        }
                        const auto check_values =
                            [&](GpuTask& candidate, const std::vector<REAL>& input, const std::vector<REAL>& expected)
                        {
                            candidate.check();
                            Symbolic_Matrix oracle(matrix);
                            for (unsigned copy = 0; copy < copies; ++copy)
                            {
                                const size_t begin = size_t(copy) * matrix.nnz;
                                compare(
                                    std::vector<REAL>(candidate.factors(copy), candidate.factors(copy) + matrix.nnz),
                                    std::vector<REAL>(expected.begin() + begin, expected.begin() + begin + matrix.nnz),
                                    "fresh input known LU");
                                oracle.val.assign(input.begin() + begin, input.begin() + begin + matrix.nnz);
                                const double residual = oracle.factor_residual(candidate.factors(copy));
                                if (!std::isfinite(residual) || residual > (sizeof(REAL) == 8 ? 1e-12 : 1e-5))
                                    throw std::runtime_error("fresh input full factor residual");
                            }
                        };
                        task.enqueue();
                        task.wait();
                        check_values(task, default_input, default_factors);
                        expect_task_rejection<std::invalid_argument>([&] { task.upload_input(nullptr, elements); },
                            "fresh input accepted null source");
                        expect_task_rejection<std::invalid_argument>([&]
                            { task.upload_input(default_input.data(), elements - 1); },
                            "fresh input accepted incorrect element count");
                        // Argument errors must preserve both usability and the previous result.
                        check_values(task, default_input, default_factors);

                        for (unsigned round = 1; round <= 3; ++round)
                        {
                            std::vector<REAL> input, expected;
                            for (unsigned copy = 0; copy < copies; ++copy)
                            {
                                const auto fresh = make_values(round, copy);
                                input.insert(input.end(), fresh.first.begin(), fresh.first.end());
                                expected.insert(expected.end(), fresh.second.begin(), fresh.second.end());
                            }
                            task.upload_input(input.data(), input.size());
                            if (round == 1)
                            {
                                expect_task_rejection<std::logic_error>([&] { task.factors(); },
                                    "upload-only fresh input allowed result access");
                                expect_task_rejection<std::logic_error>([&] { task.upload_input(); },
                                    "upload-only fresh input accepted default reupload");
                                expect_task_rejection<std::logic_error>([&]
                                    { task.upload_input(input.data(), elements); },
                                    "upload-only fresh input accepted another fresh upload");
                                expect_task_rejection<std::logic_error>([&] { task.prepare_graph(); },
                                    "upload-only fresh input allowed graph preparation");
                                expect_task_rejection<std::logic_error>([&] { task.wait(); },
                                    "upload-only fresh input allowed wait without numeric work");
                            }
                            task.enqueue();
                            if (round == 1)
                            {
                                expect_task_rejection<std::logic_error>([&] { task.factors(); },
                                    "pending fresh input allowed result access");
                                expect_task_rejection<std::logic_error>([&] { task.upload_input(); },
                                    "pending fresh input accepted default reupload");
                                expect_task_rejection<std::logic_error>([&]
                                    { task.upload_input(input.data(), elements); },
                                    "pending fresh input accepted another fresh upload");
                                expect_task_rejection<std::logic_error>([&] { task.enqueue(); },
                                    "pending fresh input allowed another enqueue");
                                expect_task_rejection<std::logic_error>([&] { task.prepare_graph(); },
                                    "pending fresh input allowed graph preparation");
                            }
                            // Keep the caller's AoS source alive and unchanged until wait.
                            task.wait();
                            check_values(task, input, expected);
                        }
                        task.enqueue();
                        task.wait();
                        check_values(task, default_input, default_factors);
                        if (copies == 3)
                        {
                            GpuMatrix perturbed(matrix, 1, true, mode, snapshot);
                            GpuTask rejected(perturbed, copies);
                            if (graph)
                                rejected.prepare_graph();
                            expect_task_rejection<std::invalid_argument>([&]
                                { rejected.upload_input(default_input.data(), elements); },
                                "perturbed task accepted fresh input");
                            rejected.enqueue();
                            rejected.wait();
                            check_values(rejected, default_input, default_factors);
                        }
                    }
                    catch (const std::exception& error)
                    {
                        std::ostringstream context;
                        context << "fresh input mode=" << unsigned(mode) << " copies=" << copies << " graph=" << graph
                                << ": " << error.what();
                        throw std::runtime_error(context.str());
                    }
                }
    std::cout << "PASS host/fresh-input-fixed-structure-known-LU-and-state-contract\n";
}

void test_compact_inputs()
{
    // One omitted input position becomes a nonzero fill value during LU.
    // Another stored input starts at zero and changes legally across rounds.
    const unsigned n = 3;
    std::vector<unsigned> columns{0, 3, 5, 8}, rows{0, 1, 2, 0, 1, 0, 1, 2};
    const std::vector<unsigned> positions{8, 0, 7, 3, 6, 4, 2, 1}; // Deliberately not sorted.
    const auto dense_values = [](unsigned round, unsigned copy)
    {
        std::vector<REAL> a(9, 0);
        for (unsigned c = 0; c < 3; ++c)
            a[c * 3 + c] = REAL(4) + REAL(round) / 8 + REAL(copy) / 64;
        a[1] = REAL(.5) + REAL(copy) / 64;
        a[2] = 1;
        a[3] = round % 2 ? REAL(0) : REAL(1);
        a[6] = REAL(1) + REAL(round) / 64;
        a[7] = round % 2 ? REAL(.0625) : REAL(0);
        return a;
    };
    const auto factor = [](std::vector<REAL> a)
    {
        for (unsigned c = 0; c < 3; ++c)
        {
            for (unsigned r = c + 1; r < 3; ++r)
                a[c * 3 + r] /= a[c * 3 + c];
            for (unsigned j = c + 1; j < 3; ++j)
                for (unsigned r = c + 1; r < 3; ++r)
                    a[j * 3 + r] -= a[c * 3 + r] * a[j * 3 + c];
        }
        return a;
    };
    const auto base = dense_values(0, 0);
    std::vector<double> initial;
    for (unsigned c = 0; c < n; ++c)
        for (unsigned p = columns[c]; p < columns[c + 1]; ++p)
            initial.push_back(base[c * 3 + rows[p]]);
    std::ostringstream out, err;
    Symbolic_Matrix matrix(n, out, err);
    matrix.fill_in(rows.data(), columns.data());
    matrix.csr();
    matrix.predictLU(rows.data(), columns.data(), initial.data());
    matrix.leveling();
    if (matrix.nnz != 9 || matrix.val != base)
        throw std::runtime_error("compact fixture must have a zero fill slot");
    for (const auto& invalid : std::vector<std::vector<unsigned>>{{}, {0, 0}, {9}})
        expect_task_rejection<std::invalid_argument>([&]
            { GpuMatrix rejected(matrix, invalid, 1, false, UpdateMode::Mapped); }, "invalid compact order accepted");
    {
        GpuMatrix ordinary(matrix, 1, false, UpdateMode::Mapped);
        GpuTask task(ordinary);
        expect_task_rejection<std::invalid_argument>([&] { task.upload_compact_input(base.data(), base.size()); },
            "compact upload without an order accepted");
    }
    for (SnapshotPolicy snapshot :
        {SnapshotPolicy::Auto, SnapshotPolicy::PerCopy, SnapshotPolicy::Shared, SnapshotPolicy::Device})
        for (UpdateMode mode : {UpdateMode::Scan, UpdateMode::Mapped, UpdateMode::Panel, UpdateMode::PanelTiled})
            for (unsigned copies : {1u, 3u, 33u})
                for (bool graph : {false, true})
                {
                    std::unique_ptr<GpuTask> task;
                    {
                        GpuMatrix ordinary(matrix, 1, false, mode, snapshot),
                            compact(matrix, positions, 1, false, mode, snapshot);
                        if (compact.compact_input_elements() != positions.size() ||
                            ordinary.compact_input_elements() != 0 ||
                            compact.shared_device_bytes() !=
                                ordinary.shared_device_bytes() + positions.size() * sizeof(unsigned) ||
                            compact.task_device_bytes(copies) !=
                                ordinary.task_device_bytes(copies) + positions.size() * copies * sizeof(REAL))
                            throw std::runtime_error("compact input payload accounting");
                        task.reset(new GpuTask(compact, copies));
                        GpuTask ordinary_task(ordinary, copies);
                        if (task->numeric_launches() != ordinary_task.numeric_launches() ||
                            task->snapshot_restore_launches() != ordinary_task.snapshot_restore_launches())
                            throw std::runtime_error("compact numeric/input layout launch accounting");
                        if (graph)
                            task->prepare_graph();
                    } // Task must retain its compact position map and default snapshot.
                    const auto check = [&](const std::vector<REAL>& inputs)
                    {
                        task->check();
                        Symbolic_Matrix oracle(matrix);
                        for (unsigned copy = 0; copy < copies; ++copy)
                        {
                            const auto first = inputs.begin() + size_t(copy) * matrix.nnz;
                            std::vector<REAL> one(first, first + matrix.nnz);
                            oracle.val = one;
                            compare(std::vector<REAL>(task->factors(copy), task->factors(copy) + matrix.nnz),
                                factor(one), "compact input independent CPU LU");
                            const double residual = oracle.factor_residual(task->factors(copy));
                            if (!std::isfinite(residual) || residual > (sizeof(REAL) == 8 ? 1e-12 : 1e-5))
                                throw std::runtime_error("compact factor residual");
                        }
                    };
                    std::vector<REAL> defaults;
                    for (unsigned copy = 0; copy < copies; ++copy)
                        defaults.insert(defaults.end(), base.begin(), base.end());
                    task->enqueue();
                    task->wait();
                    check(defaults);
                    const size_t elements = positions.size() * copies;
                    expect_task_rejection<std::invalid_argument>([&] { task->upload_compact_input(nullptr, elements); },
                        "null compact source accepted");
                    expect_task_rejection<std::invalid_argument>([&]
                        { task->upload_compact_input(defaults.data(), elements - 1); },
                        "incorrect compact count accepted");
                    check(defaults);
                    for (unsigned round = 0; round < 4; ++round)
                    {
                        std::vector<REAL> inputs, compact;
                        for (unsigned copy = 0; copy < copies; ++copy)
                        {
                            const auto a = dense_values(round, copy);
                            inputs.insert(inputs.end(), a.begin(), a.end());
                            for (unsigned p : positions)
                                compact.push_back(a[p]);
                        }
                        task->upload_compact_input(compact.data(), compact.size());
                        expect_task_rejection<std::logic_error>([&]
                            { task->upload_compact_input(compact.data(), compact.size()); },
                            "duplicate compact upload accepted");
                        expect_task_rejection<std::logic_error>([&] { task->wait(); }, "compact-only wait accepted");
                        task->enqueue();
                        expect_task_rejection<std::logic_error>([&]
                            { task->upload_compact_input(compact.data(), compact.size()); },
                            "pending compact upload accepted");
                        task->wait();
                        check(inputs);
                        if (round == 0)
                        {
                            // The full API may write a slot omitted by compact input. The
                            // next compact round must clear it instead of retaining it.
                            for (unsigned copy = 0; copy < copies; ++copy)
                                inputs[size_t(copy) * matrix.nnz + 5] = REAL(.125);
                            task->upload_input(inputs.data(), inputs.size());
                            task->enqueue();
                            task->wait();
                            check(inputs);
                        }
                    }
                    task->enqueue();
                    task->wait();
                    check(defaults);
                }
    {
        GpuMatrix compact(matrix, positions, 1, true, UpdateMode::Mapped);
        GpuTask task(compact);
        std::vector<REAL> source(positions.size(), REAL(1));
        expect_task_rejection<std::invalid_argument>([&] { task.upload_compact_input(source.data(), source.size()); },
            "perturbed compact upload accepted");
        task.enqueue();
        task.wait();
        task.check();
    }
    for (bool override_sync : {false, true})
    {
        GpuMatrix compact(matrix, positions, 1, false, UpdateMode::Mapped);
        GpuTask task(compact, 3);
        std::vector<REAL> source(positions.size() * 3, REAL(1));
        async_fault::State state;
        state.fail_upload = true;
        state.override_sync = override_sync;
        {
            async_fault::Scope scope(state);
            bool original = false;
            try
            {
                task.upload_compact_input(source.data(), source.size());
            }
            catch (const std::runtime_error& e)
            {
                original = std::string(e.what()) ==
                           std::string("upload compact matrix input: ") + cudaGetErrorString(async_fault::copy_error);
            }
            if (!original || state.h2d_calls != 1 || state.h2d_submitted != 1 || state.sync_calls != 1 ||
                state.sync_successes != 1 || !state.same_stream)
                throw std::runtime_error("compact failed upload did not drain or preserve original error");
        }
        expect_task_rejection<std::logic_error>([&] { task.enqueue(); }, "failed compact task reuse accepted");
    }
    std::cout << "PASS host/compact-input-zeros-fill-order-lifetime-and-failure-cleanup\n";
}

void test_compact_tile_tails()
{
    // Five independent 3x3 blocks: 40 stored inputs, 45 factor slots.
    // Both entry and copy dimensions cross a 32-wide tile with partial tails.
    constexpr unsigned blocks = 5, copies = 33;
    std::vector<unsigned> columns{0}, rows, positions;
    for (unsigned c = 0; c < blocks * 3; ++c)
    {
        for (unsigned r = 0; r < 3; ++r)
            if (c % 3 != 1 || r != 2)
                rows.push_back(c / 3 * 3 + r);
        columns.push_back(unsigned(rows.size()));
    }
    for (unsigned block = 0; block < blocks; ++block)
        for (unsigned p : {8u, 0u, 7u, 3u, 6u, 4u, 2u, 1u})
            positions.push_back(block * 9 + p);
    const auto values = [](unsigned round, unsigned copy)
    {
        std::vector<REAL> a(blocks * 9, 0);
        for (unsigned b = 0; b < blocks; ++b)
        {
            REAL* p = a.data() + b * 9;
            for (unsigned c = 0; c < 3; ++c)
                p[c * 3 + c] = REAL(8 + b) + REAL(round) / 8 + REAL(copy) / 128;
            p[1] = REAL(.5) + REAL(copy) / 128;
            p[2] = 1;
            p[3] = round % 2 ? REAL(0) : REAL(1);
            p[6] = REAL(1) + REAL(b) / 32;
            p[7] = round % 2 ? REAL(.0625) : REAL(0);
        }
        return a;
    };
    const auto base = values(0, 0);
    std::vector<double> initial;
    for (unsigned c = 0; c < blocks * 3; ++c)
        for (unsigned p = columns[c]; p < columns[c + 1]; ++p)
            initial.push_back(base[(c / 3) * 9 + (c % 3) * 3 + rows[p] % 3]);
    std::ostringstream out, err;
    Symbolic_Matrix matrix(blocks * 3, out, err);
    matrix.fill_in(rows.data(), columns.data());
    matrix.csr();
    matrix.predictLU(rows.data(), columns.data(), initial.data());
    matrix.leveling();
    if (matrix.nnz != 45 || matrix.val != base || positions.size() != 40)
        throw std::runtime_error("compact tile fixture layout changed");
    for (UpdateMode mode : {UpdateMode::Mapped, UpdateMode::Panel, UpdateMode::PanelTiled})
        for (bool graph : {false, true})
        {
            GpuMatrix prepared(matrix, positions, 1, false, mode);
            GpuTask task(prepared, copies);
            if (graph)
                task.prepare_graph();
            for (unsigned round = 0; round < 2; ++round)
            {
                std::vector<REAL> input;
                for (unsigned copy = 0; copy < copies; ++copy)
                {
                    const auto a = values(round, copy);
                    for (unsigned p : positions)
                        input.push_back(a[p]);
                }
                task.upload_compact_input(input.data(), input.size());
                task.enqueue();
                task.wait();
                task.check();
                Symbolic_Matrix oracle(matrix);
                for (unsigned copy = 0; copy < copies; ++copy)
                {
                    oracle.val = values(round, copy);
                    const double residual = oracle.factor_residual(task.factors(copy));
                    if (!std::isfinite(residual) || residual > (sizeof(REAL) == 8 ? 1e-12 : 1e-5))
                        throw std::runtime_error("compact entry/copy tile tail residual");
                }
            }
        }
    std::cout << "PASS host/compact-entry-and-copy-tile-tails\n";
}

void test_auto_snapshot()
{
    unsigned columns[]{0, 2, 4}, rows[]{0, 1, 0, 1};
    double input[]{4, 2, 1, 3};
    const std::vector<REAL> expected{4, REAL(.5), 1, REAL(2.5)};
    std::ostringstream out, err;
    Symbolic_Matrix matrix(2, out, err);
    matrix.fill_in(rows, columns);
    matrix.csr();
    matrix.predictLU(rows, columns, input);
    matrix.leveling();
    const std::pair<UpdateMode, SnapshotPolicy> modes[]{{UpdateMode::Scan, SnapshotPolicy::PerCopy},
        {UpdateMode::Mapped, SnapshotPolicy::PerCopy}, {UpdateMode::Panel, SnapshotPolicy::Shared},
        {UpdateMode::PanelTiled, SnapshotPolicy::PerCopy}};
    for (const auto& mode : modes)
    {
        GpuMatrix defaults(matrix, 1, false, mode.first);
        if (defaults.snapshot_policy() != SnapshotPolicy::Auto || defaults.snapshot_setup_ms() != 0)
            throw std::runtime_error("default template must request Auto without device residency");
        for (SnapshotPolicy requested :
            {SnapshotPolicy::Auto, SnapshotPolicy::PerCopy, SnapshotPolicy::Shared, SnapshotPolicy::Device})
        {
            GpuMatrix prepared(matrix, 1, false, mode.first, requested);
            if (prepared.snapshot_policy() != requested ||
                prepared.shared_device_bytes() !=
                    defaults.shared_device_bytes() +
                        (requested == SnapshotPolicy::Device ? matrix.nnz * sizeof(REAL) : 0))
                throw std::runtime_error("snapshot request or persistent payload changed");
            for (unsigned copies : {1u, 3u})
            {
                GpuTask task(prepared, copies), default_task(defaults, copies);
                const auto automatic = copies == 1 ? SnapshotPolicy::PerCopy : mode.second;
                const auto effective = requested == SnapshotPolicy::Auto ? automatic : requested;
                if (default_task.snapshot_policy() != automatic)
                    throw std::runtime_error("default template did not resolve per task geometry");
                const size_t one = matrix.nnz * sizeof(REAL);
                const size_t bytes = effective == SnapshotPolicy::Device   ? 0
                                     : effective == SnapshotPolicy::Shared ? one
                                                                           : one * copies;
                const size_t ops = effective == SnapshotPolicy::Device   ? 0
                                   : effective == SnapshotPolicy::Shared ? 1
                                                                         : copies;
                const size_t restores =
                    size_t(effective == SnapshotPolicy::Device || (effective == SnapshotPolicy::Shared && copies > 1)) +
                    size_t(copies > 1 && mode.first != UpdateMode::Scan);
                for (bool graph : {false, true})
                {
                    if (graph)
                        task.prepare_graph();
                    if (task.snapshot_policy() != effective || task.snapshot_h2d_bytes() != bytes ||
                        task.snapshot_h2d_ops() != ops || task.snapshot_restore_launches() != restores)
                        throw std::runtime_error("resolved snapshot diagnostics disagree with task policy");
                    task.enqueue();
                    task.wait();
                    task.check();
                    for (unsigned copy = 0; copy < copies; ++copy)
                        compare(std::vector<REAL>(task.factors(copy), task.factors(copy) + matrix.nnz), expected,
                            "auto snapshot known LU");
                }
            }
        }
    }
    std::cout << "PASS host/auto-snapshot-per-task-policy-override-and-graph-stability\n";
}

void test_mapped_level_grids(unsigned width, bool targets, unsigned copies, unsigned tail = 3)
{
    const unsigned n = width + tail;
    std::vector<unsigned> rows, columns{0};
    std::vector<double> input;
    for (unsigned c = 0; c < n; ++c)
    {
        for (unsigned r = 0; r < n; ++r)
            if (r == c || (c < width && r >= width) || (targets && c >= width))
            {
                rows.push_back(r);
                input.push_back(r == c ? 8.0 + r % 3 : double(int((r + 3 * c) % 9) - 4) / 512);
            }
        columns.push_back(unsigned(rows.size()));
    }
    std::ostringstream out, err;
    Symbolic_Matrix matrix(n, out, err);
    matrix.fill_in(rows.data(), columns.data());
    matrix.csr();
    matrix.predictLU(rows.data(), columns.data(), input.data());
    matrix.leveling();
    if (matrix.level_ptr[1] != int(width))
        throw std::runtime_error("unexpected mapped grid test structure");
    GpuMatrix gpu(matrix, 32, false, UpdateMode::Mapped);
    GpuTask task(gpu, copies);
    const unsigned updates = targets ? tail * width : 0;
    unsigned expected = 0;
    if (width > 16)
        expected = copies > 1 ? width + updates : width <= 64 ? updates : width;
#if !defined(GLU_USE_HIP) && !defined(GLU_USE_FP32)
    if (copies == 1 && width > 16 && width <= 64 && updates * tail >= 100000)
        expected += width;
#endif
    if (task.numeric_wide_blocks() != expected)
        throw std::runtime_error("incorrect mapped level grid");
    task.prepare_graph();
    task.enqueue();
    task.wait();
    task.check();
    for (unsigned copy = 0; copy < copies; ++copy)
        if (!(matrix.factor_residual(task.factors(copy)) <= (sizeof(REAL) == 8 ? 2.e-12 : 2.e-5)))
            throw std::runtime_error("mapped level grid factor residual");
}

void test_mapped_single_normalization(unsigned lower_count, unsigned target_count, bool tail, bool perturb)
{
    // Root 0 updates the full rectangle. Root 1 shares its level, has a
    // non-unit pivot and a lower entry, but never contributes to a target.
    const unsigned first_target = 2 + lower_count, aux = first_target + target_count;
    const unsigned n = aux + (tail ? 3 : 1);
    std::vector<unsigned> rows, columns{0};
    std::vector<double> input;
    const auto add = [&](unsigned r, double v) {
        rows.push_back(r);
        input.push_back(v);
    };
    for (unsigned c = 0; c < n; ++c)
    {
        if (c >= first_target && c < aux)
            add(0, c == first_target ? 8 : 0);
        double pivot = 4;
        if (c == 1)
            pivot = perturb ? 0 : 8;
        if (tail && c == aux)
            pivot = 6;
        if (tail && c == aux + 1)
            pivot = 8;
        add(c, pivot);
        if (c == 0)
            for (unsigned r = 2; r < first_target; ++r)
                add(r, r == 2 ? 2 : 0);
        if (c == 1)
            add(aux, 4);
        if (tail && c == aux)
            add(aux + 1, 3);
        if (tail && c == aux + 1)
            add(aux + 2, 4);
        columns.push_back(unsigned(rows.size()));
    }
    std::ostringstream out, err;
    Symbolic_Matrix matrix(n, out, err);
    matrix.fill_in(rows.data(), columns.data());
    matrix.csr();
    matrix.predictLU(rows.data(), columns.data(), input.data());
    matrix.leveling();
    if (matrix.num_lev != (tail ? 4 : 2) || matrix.level_ptr[1] != 2)
        throw std::runtime_error("unexpected mapped normalization fixture levels");
#if defined(GLU_USE_HIP)
    constexpr bool cuda_fp64 = false;
#else
    constexpr bool cuda_fp64 = sizeof(REAL) == 8;
#endif
    const bool heavy = lower_count * target_count >= 100000;
    GpuMatrix gpu(matrix, 32, perturb, UpdateMode::Mapped);
    // Only the zero diagonal at root 1 is perturbed. The maximum column sum is 12.
    Symbolic_Matrix oracle(matrix);
    const REAL pivot1 = perturb ? REAL(12) * REAL(3.45e-4f) : REAL(8);
    oracle.val[matrix.l_col_ptr[1]] = pivot1;
    const auto schur = matrix.l_col_ptr[first_target] - lower_count;
    if (matrix.sym_r_idx[schur] != 2)
        throw std::runtime_error("unexpected mapped normalization Schur entry");
    for (unsigned copies : {1u, 3u})
    {
        GpuTask task(gpu, copies);
        // AoS has a root update, one wide level and a final normalization.
        // Early root normalization adds one launch and can remove the final one.
        const unsigned aos = 3 + unsigned(cuda_fp64 && heavy && tail);
        // SoA also converts the output. CUDA FP64 normalizes each of the two
        // optional singleton levels separately, other builds use one final list.
        const unsigned soa = 4 + (cuda_fp64 && tail ? 2 : 0);
        if (task.numeric_launches() != (copies == 1 ? aos : soa) ||
            task.normalizes_each_level() != (copies > 1 && cuda_fp64))
            throw std::runtime_error("incorrect mapped normalization schedule");
        for (bool graph : {false, true})
        {
            if (graph)
                task.prepare_graph();
            task.enqueue();
            task.wait();
            task.check();
            for (unsigned copy = 0; copy < copies; ++copy)
            {
                const auto* lu = task.factors(copy);
                const REAL root1_lower = lu[matrix.l_col_ptr[1] + 1];
                if (lu[matrix.l_col_ptr[0] + 1] != REAL(0.5) || lu[schur] != REAL(-4) ||
                    lu[matrix.l_col_ptr[1]] != pivot1 ||
                    std::abs(root1_lower * pivot1 - REAL(4)) > 8 * std::numeric_limits<REAL>::epsilon() ||
                    (tail && lu[matrix.l_col_ptr[aux + 1] + 1] != REAL(0.5)))
                    throw std::runtime_error("mapped normalization pivot, raw lower or Schur entry");
                if (!(oracle.factor_residual(lu) <= (sizeof(REAL) == 8 ? 2.e-12 : 2.e-5)))
                    throw std::runtime_error("mapped normalization factor residual");
            }
        }
    }
}

void test_mapped_front(unsigned steps, unsigned copies, bool perturb, bool graph)
{
    // Interleaved exterior rows and columns form a rectangular front whose
    // entries occupy different sparse slots in each target column.
    const unsigned n = steps + 13;
    std::vector<unsigned> rows, columns{0};
    std::vector<double> input;
    for (unsigned c = 0; c < n; ++c)
    {
        for (unsigned r = 0; r < n; ++r)
        {
            const bool front_row = r < steps || (r - steps) % 2 == 0;
            const bool front_column = c < steps || (c - steps) % 2 == 1;
            if (r != c && !(front_row && front_column))
                continue;
            rows.push_back(r);
            const double diagonal = perturb && r == 17 ? 1.e-7 : 8.0 + c % 3;
            input.push_back(r == c ? diagonal : double(int((7 * r + 11 * c) % 17) - 8) / 4096);
        }
        columns.push_back(unsigned(rows.size()));
    }
    // U(2,6) = 1 - round((1 + eps) * (1 - eps)). A contracted
    // multiply-add would leave eps^2. The second source has L(2,1) = 0.
    const double eps = std::numeric_limits<REAL>::epsilon();
    input[0] = 1;
    input[2] = 1 + eps;
    input[columns[1]] = input[columns[1] + 2] = 0;
    input[columns[6]] = 1 - eps;
    input[columns[6] + 2] = 1;
#if defined(GLU_USE_FP32)
    // U(1,2..5) each receives just one update with L(1,0) = 1. These
    // distinguish flushing either sign of the sum, the initial value and the addend.
    const double minimum = std::numeric_limits<float>::min();
    input[1] = 1;
    input[columns[2]] = minimum;
    input[columns[2] + 1] = minimum + std::numeric_limits<float>::denorm_min();
    input[columns[3]] = minimum;
    input[columns[3] + 1] = minimum / 2;
    input[columns[4]] = -minimum / 2;
    input[columns[4] + 1] = minimum;
    input[columns[5]] = minimum + std::numeric_limits<float>::denorm_min();
    input[columns[5] + 1] = minimum;
#endif
    std::ostringstream out, err;
    Symbolic_Matrix matrix(n, out, err);
    matrix.fill_in(rows.data(), columns.data());
    matrix.csr();
    matrix.predictLU(rows.data(), columns.data(), input.data());
    matrix.leveling();
    GpuMatrix mapped(matrix, 32, perturb, UpdateMode::Mapped), scan(matrix, 32, perturb, UpdateMode::Scan);
    GpuTask task(mapped, copies), reference(scan, copies);
    const size_t ordinary_bytes = size_t(matrix.nnz) * (copies + task.soa_value_stride()) * sizeof(REAL);
    const bool blocked = steps >= 32 && copies == 1;
    if ((mapped.task_device_bytes(copies) > ordinary_bytes) != blocked || (blocked && task.numeric_launches() >= steps))
        throw std::runtime_error("incorrect mapped front workspace or launch count");
    if (graph)
    {
        task.prepare_graph();
        reference.prepare_graph();
    }
    std::vector<REAL> fresh(size_t(copies) * matrix.nnz);
    for (unsigned copy = 0; copy < copies; ++copy)
        for (unsigned i = 0; i < matrix.nnz; ++i)
            fresh[size_t(copy) * matrix.nnz + i] = matrix.val[i] * (REAL(1) + REAL(copy + 1) / 8 + REAL(i % 5) / 64);
    for (bool new_values : {false, true, false})
    {
        if (new_values)
        {
            if (perturb)
                continue;
            task.upload_input(fresh.data(), fresh.size());
            reference.upload_input(fresh.data(), fresh.size());
        }
        task.enqueue();
        reference.enqueue();
        task.wait();
        reference.wait();
        task.check();
        reference.check();
        for (unsigned copy = 0; copy < copies; ++copy)
        {
            const auto* factors = task.factors(copy);
            const auto* other = reference.factors(copy);
            compare({factors, factors + matrix.nnz}, {other, other + matrix.nnz}, "mapped rectangular front vs Scan");
            const unsigned rounded = matrix.sym_c_ptr[6] + 2;
            if (!new_values && (other[rounded] != 0 || factors[rounded] != 0))
                throw std::runtime_error("mapped front must round the product before subtracting at U(2,6)");
#if defined(GLU_USE_FP32)
            for (unsigned c = 2; c <= 5; ++c)
            {
                const unsigned position = matrix.sym_c_ptr[c] + 1;
                if (std::memcmp(factors + position, other + position, sizeof(REAL)) != 0)
                    throw std::runtime_error("mapped front differs from sparse FP32 atomic add at U(1," +
                                             std::to_string(c) + ")");
            }
#endif
            if (!perturb)
            {
                Symbolic_Matrix oracle(matrix);
                if (new_values)
                    std::copy_n(fresh.data() + size_t(copy) * matrix.nnz, matrix.nnz, oracle.val.begin());
                if (!(oracle.factor_residual(factors) <= (sizeof(REAL) == 8 ? 2.e-12 : 2.e-5)))
                    throw std::runtime_error("mapped rectangular front factor residual");
            }
        }
    }
}

enum class SplitFixture
{
    independent,
    duplicate_lower,
    duplicate_upper,
    pending_parent
};

constexpr unsigned split_first_row = 66, split_exterior_size = 192;
constexpr unsigned split_first_target = split_first_row + split_exterior_size;
constexpr unsigned split_order = split_first_target + split_exterior_size;

void make_split_fixture(Symbolic_Matrix& a, SplitFixture fixture)
{
    // Pivot chains [0,4..34] and [2,35..65] share a large exterior rectangle,
    // with all but its first row and column structurally present but zero.
    // The sparse roots 1 and 3 also update the first exterior entry.
    // Original level zero is [0,1,2,3], so two selected fronts leave disjoint
    // remainder intervals [1,2) and [3,4).
    const auto left = [](unsigned k) { return k == 0 || (k >= 4 && k <= 34); };
    const auto right = [](unsigned k) { return k == 2 || (k >= 35 && k <= 65); };
    const bool blocker = fixture == SplitFixture::pending_parent;
    std::vector<unsigned> rows, columns{0};
    std::vector<double> input;
    for (unsigned c = 0; c < a.n; ++c)
    {
        for (unsigned r = 0; r < a.n; ++r)
        {
            const bool exterior_row = r >= split_first_row && r < split_first_target;
            if (r != c && !(left(c) && left(r)) && !(right(c) && right(r)) &&
                !(exterior_row && c < split_first_row) && !(c >= split_first_target && r < split_first_target) &&
                !(blocker && ((r == 4 && c == 1) || (r == 1 && c == 4))))
                continue;
            // The two dense roots have different non-unit divisors. Both raw
            // lower entries must be normalized exactly once at factor output.
            const double pivot = c == 0 ? 2.0 : (c == 2 || (blocker && c == 4)) ? 4.0 : 1.0;
            double v = r == c ? pivot : 0.0;
            if (r == split_first_row && c < 4)
                v = pivot;
            if (c == split_first_target && r < 4)
                v = 1.0;
            if (r == split_first_row && c == split_first_target)
                v = 8.0;
            if (blocker && ((r == 4 && c == 1) || (r == 1 && c == 4)))
                v = 1.0;
            const bool duplicate = (fixture == SplitFixture::duplicate_lower && r == split_first_row && left(c)) ||
                                   (fixture == SplitFixture::duplicate_upper && c == split_first_target && left(r));
            rows.push_back(r);
            input.push_back(v);
            if (duplicate)
            {
                rows.push_back(r);
                input.push_back(v);
            }
        }
        columns.push_back(unsigned(rows.size()));
    }
    a.fill_in(rows.data(), columns.data());
    a.csr();
    a.predictLU(rows.data(), columns.data(), input.data());
    a.leveling();
}

void test_split_fixture_plan(const Symbolic_Matrix& a, SplitFixture fixture)
{
    if (a.csr_diag_ptr.size() != a.n || a.level_ptr[1] != 4 ||
        !std::equal(a.level_idx.begin(), a.level_idx.begin() + 4, std::vector<int>{0, 1, 2, 3}.begin()))
        throw std::runtime_error("split fixture initial level or diagonal count");
    std::vector<uint64_t> bases{0};
    for (unsigned c = 0; c < a.n; ++c)
        bases.push_back(bases.back() + uint64_t(a.sym_c_ptr[c + 1] - a.l_col_ptr[c] - 1) *
                                           (a.csr_r_ptr[c + 1] - a.csr_diag_ptr[c] - 1));
    MappedFrontPlan plan;
    std::vector<int> mapped_order = a.level_idx;
    add_split_mapped_levels(plan, a, bases, mapped_order);
    const auto& first = plan.split_levels.at(0);
    const bool both = fixture == SplitFixture::independent;
    if (first.fronts.size() != (both ? 2u : 1u))
        throw std::runtime_error("split fixture accepted incorrect number of fronts");
    for (unsigned i = 0; i < first.fronts.size(); ++i)
    {
        const auto& f = first.fronts[i];
        const unsigned root = both && i == 0 ? 0 : 2;
        if (f.steps != 32 || f.rows != 32 + split_exterior_size || f.cols != 32 + split_exterior_size ||
            f.diagonal != a.l_col_ptr[root] ||
            f.first_target != a.csr_diag_ptr[root] + 1 || f.map_begin != bases[root])
            throw std::runtime_error("split fixture selected incorrect branch");
    }
    if (!std::equal(a.level_idx.begin(), a.level_idx.end(), mapped_order.begin()))
        throw std::runtime_error("split fixture changed the original source order");
    std::vector<int> expected = both ? std::vector<int>{1, 3} : std::vector<int>{0, 1, 3};
    const auto rem = first.sparse;
    if (rem.begin < a.n || rem.end > mapped_order.size() ||
        !std::equal(expected.begin(), expected.end(), mapped_order.begin() + rem.begin, mapped_order.begin() + rem.end))
        throw std::runtime_error("split fixture changed sparse remainder identities");
    for (unsigned lev = 1; lev < plan.split_levels.size(); ++lev)
        if (!plan.split_levels[lev].fronts.empty())
            throw std::runtime_error("split fixture scheduled an unexpected later root");
}

void test_mapped_split_front(SplitFixture fixture, bool graph)
{
    std::ostringstream out, err;
    Symbolic_Matrix matrix(split_order, out, err);
    make_split_fixture(matrix, fixture);
    test_split_fixture_plan(matrix, fixture);
    GpuMatrix storage(matrix, 32, false, UpdateMode::Mapped);
    GpuTask task(storage, 1);
    if (graph)
        task.prepare_graph();
    task.enqueue();
    task.wait();
    task.check();
    const auto* values = task.factors(0);
    const auto value = [&](unsigned row, unsigned col)
    {
        const auto begin = matrix.sym_r_idx.begin() + matrix.sym_c_ptr[col];
        const auto end = matrix.sym_r_idx.begin() + matrix.sym_c_ptr[col + 1];
        const auto position = std::find(begin, end, row);
        if (position == end)
            throw std::runtime_error("split fixture missing factor entry");
        return values[position - matrix.sym_r_idx.begin()];
    };
    const bool blocked = fixture == SplitFixture::pending_parent;
    const REAL expected = fixture == SplitFixture::independent ? REAL(4) : blocked ? REAL(11) / 3 : REAL(3);
    if (std::abs(value(split_first_row, split_first_target) - expected) > 4 * std::numeric_limits<REAL>::epsilon() ||
        (blocked && value(4, 4) != REAL(3)) || value(split_first_row, 0) != REAL(1) ||
        value(split_first_row, 2) != REAL(1))
        throw std::runtime_error("split mapped front numeric result");
    if ((fixture == SplitFixture::independent || blocked) &&
        !(matrix.factor_residual(values) <= (sizeof(REAL) == 8 ? 2.e-12 : 2.e-5)))
        throw std::runtime_error("split mapped front factor residual");
}

void test_invalid_factor(REAL value, const char* diagnostic)
{
    unsigned column[]{0, 1}, row[]{0};
    double values[]{static_cast<double>(value)};
    std::ostringstream out, err;
    Symbolic_Matrix matrix(1, out, err);
    matrix.fill_in(row, column);
    matrix.csr();
    matrix.predictLU(row, column, values);
    matrix.leveling();
    if (LUonDevice(matrix, out, err, false) || err.str().find(diagnostic) == std::string::npos)
        throw std::runtime_error(std::string("missing failure: ") + diagnostic + ": " + err.str());
    std::cout << "PASS host/" << diagnostic << '\n';
}

} // namespace

void test_local_deferred_group_order()
{
    auto same = [](const auto& a, const auto& b)
    { return a.size() == b.size() && (a.empty() || !std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0]))); };
    // Two disjoint deferred ranges, with immediate groups before, between and
    // after them. Vary prefixes and tails independently of the eight-target grid.
    for (unsigned prefix = 0; prefix < 10; ++prefix)
        for (unsigned count = 0; count < 20; ++count)
            for (unsigned pattern = 0; pattern < 4; ++pattern)
            {
                const unsigned second = prefix + count + 3, last = second + (count + 5) % 17, total = last + 2;
                std::vector<unsigned> lengths(total), expected_order;
                for (unsigned i = 0; i < total; ++i)
                    lengths[i] = pattern == 0   ? 1
                                 : pattern == 1 ? 4
                                                : (i % 3 == 0        ? 1
                                                      : pattern == 2 ? 2 + (total - i) % 4
                                                                     : 2 + (i * 7) % 11);
                // Ordinary, fully inlined and mixed singleton representations must all
                // produce the same logical target order without changing contribution order.
                for (unsigned format = 0; format < 3; ++format)
                {
                    PanelUpdatePlan plan;
                    plan.immediate = {{0, prefix}, {prefix + count, second}, {last, total}};
                    plan.deferred = {{prefix, prefix + count}, {second, last}, {total, total}};
                    plan.dense_ranges = {{0, 0}, {0, 1}, {1, 1}};
                    plan.dense_tiles = {{0, 2, 4, 1, 1, 2, 0, 0}};
                    plan.dense_indices = {7, 0, 8, 9, 10, 11};
                    plan.dense_products = 2;
                    for (unsigned i = 0; i < total; ++i)
                    {
                        const unsigned destination = i + 1;
                        const bool single = lengths[i] == 1 && (format == 1 || (format == 2 && i % 2));
                        if (single)
                        {
                            plan.groups.push_back(
                                {destination | inline_update_pair_bit, 100 + 2 * destination, 10 + destination});
                            ++plan.inline_singletons;
                        }
                        else
                        {
                            const unsigned begin = unsigned(plan.pairs.size());
                            for (unsigned q = 0; q < lengths[i]; ++q)
                                plan.pairs.push_back({100 + 2 * destination + q, 10 + destination + q});
                            plan.groups.push_back({destination, begin, unsigned(plan.pairs.size())});
                        }
                        if ((i >= prefix && i < prefix + count) || (i >= second && i < last))
                            plan.deferred_products += lengths[i];
                        else
                            plan.immediate_products += lengths[i];
                    }
                    plan.deferred_products += plan.dense_products;
                    const auto before = plan;
                    sort_deferred_groups_locally(plan, format != 0);
                    if (!same(plan.pairs, before.pairs) || !same(plan.immediate, before.immediate) ||
                        !same(plan.deferred, before.deferred) || !same(plan.dense_ranges, before.dense_ranges) ||
                        !same(plan.dense_tiles, before.dense_tiles) || plan.dense_indices != before.dense_indices ||
                        plan.immediate_products != before.immediate_products ||
                        plan.deferred_products != before.deferred_products ||
                        plan.dense_products != before.dense_products ||
                        plan.inline_singletons != before.inline_singletons ||
                        plan.groups.size() != before.groups.size())
                        throw std::runtime_error("local group order changed plan payload or accounting");
                    std::vector<unsigned> order;
                    std::vector<bool> seen(total, false);
                    for (unsigned i = 0; i < total; ++i)
                    {
                        const auto g = plan.groups[i];
                        const unsigned source = (g.destination & ~inline_update_pair_bit) - 1;
                        if (source >= total || seen[source] || std::memcmp(&g, &before.groups[source], sizeof(g)))
                            throw std::runtime_error("local group order lost or changed a target contribution tuple");
                        seen[source] = true;
                        order.push_back(source);
                        unsigned begin = total, end = total;
                        for (const auto r : before.deferred)
                            if (i >= r.begin && i < r.end)
                            {
                                begin = r.begin;
                                end = r.end;
                            }
                        if (begin == total || lengths[source] == 1)
                        {
                            if (source != i)
                                throw std::runtime_error("local group order moved an immediate or singleton slot");
                        }
                        else if (source < begin || source >= end || (source - begin) / 8 != (i - begin) / 8)
                            throw std::runtime_error("local group order crossed a range or eight-target boundary");
                    }
                    for (const auto r : plan.deferred)
                        for (unsigned base = r.begin; base < r.end; base += 8)
                        {
                            unsigned previous = total;
                            for (unsigned i = base; i < std::min(base + 8, r.end); ++i)
                                if (lengths[order[i]] > 1)
                                {
                                    const unsigned source = order[i];
                                    if (previous != total &&
                                        (lengths[previous] > lengths[source] ||
                                            (lengths[previous] == lengths[source] && previous > source)))
                                        throw std::runtime_error(
                                            "local group order is not stable by contribution count");
                                    previous = source;
                                }
                        }
                    if (!format)
                        expected_order = order;
                    else if (order != expected_order)
                        throw std::runtime_error("local group order depends on singleton representation");
                    const auto ordered = plan.groups;
                    sort_deferred_groups_locally(plan, format != 0);
                    if (!same(plan.groups, ordered))
                        throw std::runtime_error("local group order is not idempotent");
                }
            }
    // A high destination bit is data when ordinary groups cannot use inline tags.
    PanelUpdatePlan wide;
    wide.groups = {{inline_update_pair_bit, 0, 4}, {inline_update_pair_bit + 1, 4, 6},
        {inline_update_pair_bit + 2, 6, 7}};
    wide.deferred = {{0, 3}};
    sort_deferred_groups_locally(wide, false);
    if (wide.groups[0].destination != inline_update_pair_bit + 1 ||
        wide.groups[1].destination != inline_update_pair_bit ||
        wide.groups[2].destination != inline_update_pair_bit + 2)
        throw std::runtime_error("local group order mistook an ordinary destination for an inline tag");
    std::cout << "PASS panel/local-deferred-order-contributions-singletons-ranges-and-representations\n";
}

void test_packed_update_groups_codec()
{
    static_assert(packed_update_single_bit == (uint64_t(1) << 63) && packed_update_position_mask == ((1u << 21) - 1) &&
                      packed_update_pair_mask == ((1u << 27) - 1) && packed_update_count_mask == ((1u << 15) - 1),
        "packed update field widths changed");
    auto range = [](uint64_t record, unsigned destination, unsigned begin, unsigned count)
    {
        if ((record & (uint64_t(1) << 63)) || unsigned(record & ((uint64_t(1) << 21) - 1)) != destination ||
            unsigned((record >> 21) & ((uint64_t(1) << 27) - 1)) != begin || unsigned(record >> 48) != count)
            throw std::runtime_error("packed range field boundary");
    };
    const unsigned pair_limit = 1u << 27;
    range(pack_update_range(packed_update_position_mask, packed_update_pair_mask, 1), packed_update_position_mask,
        packed_update_pair_mask, 1);
    range(
        pack_update_range(packed_update_position_mask, pair_limit - packed_update_count_mask, packed_update_count_mask),
        packed_update_position_mask, pair_limit - packed_update_count_mask, packed_update_count_mask);
    auto rejects = [](unsigned destination, unsigned begin, unsigned count)
    {
        bool rejected = false;
        try
        {
            (void)pack_update_range(destination, begin, count);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("packed range accepted an invalid field");
    };
    rejects(packed_update_position_mask + 1, 0, 1);
    rejects(0, packed_update_pair_mask + 1, 1);
    rejects(0, 0, 0);
    rejects(0, 0, packed_update_count_mask + 1);
    rejects(0, packed_update_pair_mask, 2);
    // Exercise 32-bit pair positions without allocating a 2^27-entry vector.
    rejects(0, 1u << 31, 2);
    rejects(0, std::numeric_limits<unsigned>::max(), 2);

    const unsigned top = packed_update_position_mask - 1;
    PanelUpdatePlan ordinary;
    ordinary.groups = {{top, 0, 1}};
    ordinary.pairs = {{top, top - 1}};
    const uint64_t expected = (uint64_t(1) << 63) | uint64_t(top) | (uint64_t(top) << 21) | (uint64_t(top - 1) << 42);
    const auto first = pack_update_groups(ordinary, packed_update_position_mask, false);
    PanelUpdatePlan inlined;
    inlined.groups = {{inline_update_pair_bit | top, top, top - 1}};
    const auto second = pack_update_groups(inlined, packed_update_position_mask, true);
    if (first.size() != 1 || first[0] != expected || second != first)
        throw std::runtime_error("packed ordinary/inline singleton boundary");
    if (!pack_update_groups(ordinary, 1u << 21, false).empty() || ordinary.groups.size() != 1 ||
        ordinary.groups[0].destination != top || ordinary.groups[0].begin != 0 || ordinary.groups[0].end != 1 ||
        ordinary.pairs.size() != 1 || ordinary.pairs[0].lower != top || ordinary.pairs[0].pivot != top - 1)
        throw std::runtime_error("packed nnz fallback changed the original plan");

    PanelUpdatePlan multiple;
    multiple.groups = {{1, 0, 3}};
    multiple.pairs = {{2, 3}, {4, 5}, {6, 7}};
    const auto third = pack_update_groups(multiple, 8, false);
    if (third.size() != 1)
        throw std::runtime_error("packed multi record count");
    range(third[0], 1, 0, 3);
    const NormalizedUpdatePair original_pairs[] = {{2, 3}, {4, 5}, {6, 7}};
    if (multiple.groups[0].destination != 1 || multiple.groups[0].begin != 0 || multiple.groups[0].end != 3 ||
        std::memcmp(multiple.pairs.data(), original_pairs, sizeof(original_pairs)))
        throw std::runtime_error("packed multi contribution order changed");

    // The preceding singleton must not leave a partially packed template when
    // a real, in-bounds multi group exceeds the fixed 15-bit count limit.
    PanelUpdatePlan excessive;
    excessive.pairs.resize(size_t(packed_update_count_mask) + 1, NormalizedUpdatePair{0, 0});
    excessive.groups = {{inline_update_pair_bit | 1u, 0, 0}, {0, 0, packed_update_count_mask + 1}};
    if (!pack_update_groups(excessive, 2, true).empty() || excessive.groups.size() != 2 ||
        excessive.groups[0].destination != (inline_update_pair_bit | 1u) || excessive.groups[1].begin != 0 ||
        excessive.groups[1].end != packed_update_count_mask + 1 ||
        excessive.pairs.size() != size_t(packed_update_count_mask) + 1)
        throw std::runtime_error("packed long-group fallback changed the original plan");
    excessive.pairs.pop_back();
    --excessive.groups[1].end;
    const auto maximum = pack_update_groups(excessive, 2, true);
    if (maximum.size() != 2 || maximum[0] != (packed_update_single_bit | 1u))
        throw std::runtime_error("packed maximum-length template was rejected");
    range(maximum[1], 0, 0, packed_update_count_mask);
    PanelUpdatePlan invalid;
    invalid.groups = {{0, 0, 2}};
    invalid.pairs = {{0, 0}};
    bool rejected = false;
    try
    {
        (void)pack_update_groups(invalid, 1, false);
    }
    catch (const std::logic_error&)
    {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("packed invalid pair range was silently accepted");
    std::cout << "PASS panel/packed-codec-boundaries-singletons-and-template-fallback\n";
}

void test_inline_update_groups()
{
    Stream stream;
    for (unsigned copies : {1u, 3u, 33u})
        for (bool nonfinite : {false, true})
        {
            const unsigned bits = filled_copy_bits(copies, 5), lanes = 1u << bits;
            const unsigned stride = (copies + lanes - 1) / lanes * lanes;
            std::vector<REAL> input(size_t(18) * stride, REAL(123.5));
            for (unsigned copy = 0; copy < copies; ++copy)
            {
                auto at = [&](unsigned position) -> REAL& { return input[size_t(position) * stride + copy]; };
                at(0) = 10;
                at(1) = 20;
                at(2) = 30;
                at(8) = REAL(2) + REAL(copy) / 8;
                at(9) = REAL(.5);
                // The small final term is lost if it is moved before cancellation.
                const REAL large = std::ldexp(REAL(1), std::numeric_limits<REAL>::digits + 8);
                at(10) = large;
                at(11) = 1;
                at(12) = -large;
                at(13) = 1;
                at(16) = 1;
                at(17) = 1;
                at(14) = nonfinite && copy % 2 == 0 ? REAL(0) : REAL(4);
                at(15) = nonfinite && copy % 2 == 0 ? std::numeric_limits<REAL>::infinity() : REAL(.125);
            }
            for (unsigned format : {0u, 1u, 2u})
            {
                const bool compact = format != 0, packed = format == 2;
                std::vector<UpdateContribution> ops{{0, 8, 9}, {1, 10, 11}, {1, 12, 13}, {1, 16, 17}, {2, 14, 15}};
                std::vector<UpdateGroup> groups{{0, 0, 0}};
                std::vector<NormalizedUpdatePair> pairs{{0, 0}};
                append_update_groups(ops, groups, pairs, compact);
                if (groups.size() != 4 || pairs.size() != (compact ? 4u : 6u))
                    throw std::runtime_error("inline group storage or prefix changed");
                std::vector<uint64_t> records(1, 0);
                if (packed)
                {
                    PanelUpdatePlan plan;
                    // The fake group prefix is not a valid contribution group.
                    // Keep the real pair prefix so begin offsets remain nonzero.
                    plan.groups.assign(groups.begin() + 1, groups.end());
                    plan.pairs = pairs;
                    records = pack_update_groups(plan, 18, compact);
                    if (records.size() != 3)
                        throw std::runtime_error("packed group oracle unexpectedly fell back");
                    records.insert(records.begin(), 0);
                }
                DeviceBuffer<REAL> values(input);
                DeviceBuffer<UpdateGroup> device_groups(groups);
                DeviceBuffer<NormalizedUpdatePair> device_pairs(pairs);
                DeviceBuffer<uint64_t> device_records(records);
                if (packed)
                    packed_group_update<<<dim3(1, (copies + lanes - 1) / lanes), 256, 0, stream.value>>>(values.data(),
                        copies, stride, bits, device_records.data(), device_pairs.data(), 1, 4);
                else
                    mapped_group_update<<<dim3(1, (copies + lanes - 1) / lanes), 256, 0, stream.value>>>(values.data(),
                        copies, stride, bits, device_groups.data(), device_pairs.data(), 1, 4, compact);
                check_gpu(cudaGetLastError(), "launch inline group oracle");
                check_gpu(cudaStreamSynchronize(stream.value), "wait inline group oracle");
                const auto actual = values.read();
                for (unsigned position = 0; position < 18; ++position)
                    for (unsigned copy = 0; copy < stride; ++copy)
                    {
                        const size_t at = size_t(position) * stride + copy;
                        if (position < 3 && copy < copies)
                        {
                            REAL expected = input[at];
                            for (const auto& u : ops)
                                if (u.destination == position)
                                {
                                    const REAL factor = input[size_t(u.lower) * stride + copy];
                                    if (factor != REAL(0))
                                        expected -= factor * input[size_t(u.pivot) * stride + copy];
                                }
                            if (!std::isfinite(actual[at]) || actual[at] != expected)
                                throw std::runtime_error("inline group oracle or zero-factor semantics");
                        }
                        else if (std::memcmp(&actual[at], &input[at], sizeof(REAL)) != 0)
                        {
                            throw std::runtime_error("inline group operand or padding changed");
                        }
                    }
            }
        }
    std::cout << "PASS panel/ordinary-inline-packed-groups-order-prefix-tail-and-nonfinite\n";
}

int main()
{
    try
    {
        GpuMatrix::select_device();
        cudaDeviceProp properties;
        check_gpu(cudaGetDeviceProperties(&properties, 0), "inspect device");
        std::cout << "Device: " << properties.name << ", warp size: " << properties.warpSize
                  << ", REAL bits: " << sizeof(REAL) * 8 << '\n';
        for (bool perturb : {false, true})
        {
            const Matrix matrix = make_matrix(perturb);
            for (unsigned threads : {64u, 128u, 1024u})
                test_batched(matrix, threads, perturb,
                    std::string(perturb ? "RL_perturb/" : "RL/") + std::to_string(threads));
            for (unsigned threads : {256u, 1024u})
                test_one_column(matrix, perturb,
                    std::string(perturb ? "one-column/perturb/" : "one-column/plain/") + std::to_string(threads),
                    threads);
        }
        const Matrix empty_lower = make_empty_lower_matrix();
        test_batched(empty_lower, 64, true, "RL_perturb/empty-lower");
        test_one_column(empty_lower, true, "one-column/empty-lower");
        test_limited_slots();
        for (unsigned width : {16u, 17u, 64u, 65u})
            for (bool targets : {false, true})
                for (unsigned copies : {1u, 3u})
                    test_mapped_level_grids(width, targets, copies);
        test_mapped_level_grids(40, true, 1, 49); // 96,040 products
        test_mapped_level_grids(40, true, 1, 50); // 100,000 products
        for (bool perturb : {false, true})
        {
            test_mapped_single_normalization(271, 369, true, perturb); // 99,999 products
            test_mapped_single_normalization(250, 400, true, perturb); // 100,000 products, deferred tail
            test_mapped_single_normalization(250, 400, false, perturb); // 100,000 products, empty deferred list
        }
        for (bool graph : {false, true})
            for (bool perturb : {false, true})
            {
                test_mapped_front(31, 1, perturb, graph);
                test_mapped_front(32, 1, perturb, graph);
                test_mapped_front(39, 1, perturb, graph);
                test_mapped_front(39, 3, perturb, graph);
            }
        for (bool graph : {false, true})
            for (SplitFixture fixture : {SplitFixture::independent, SplitFixture::duplicate_lower,
                     SplitFixture::duplicate_upper, SplitFixture::pending_parent})
                test_mapped_split_front(fixture, graph);
        for (UpdateMode mode : {UpdateMode::Scan, UpdateMode::Mapped})
            for (bool graph : {false, true})
                for (unsigned copies : {1u, 2u, 3u, 16u, 33u, 42u})
                    test_repeated_tasks(graph, copies, mode);
        for (bool graph : {false, true})
            for (unsigned copies : {1u, 2u, 3u, 33u, 42u})
                for (bool perturb : {false, true})
                    test_small_level(graph, copies, perturb);
        for (unsigned copies : {1u, 2u, 3u, 7u, 8u, 9u, 31u, 32u, 33u, 42u})
            for (unsigned variant = 0; variant < 5; ++variant)
                for (bool perturb : {false, true})
                    for (unsigned path = 0; path < 3; ++path)
                        test_layout(copies, variant, perturb, path);
        test_dense_updates();
        test_local_deferred_group_order();
        test_packed_update_groups_codec();
        test_inline_update_groups();
        test_panel_boundaries();
        test_fused_ordinary_groups();
        test_fused_strip_operands();
        for (SnapshotPolicy snapshot : {SnapshotPolicy::Shared, SnapshotPolicy::Device})
            for (UpdateMode mode : {UpdateMode::Scan, UpdateMode::Mapped})
                for (unsigned copies : {1u, 3u, 33u})
                    for (bool graph : {false, true})
                        test_repeated_tasks(graph, copies, mode, snapshot);
        test_fresh_inputs();
        test_compact_inputs();
        test_compact_tile_tails();
        test_auto_snapshot();
        test_submission_failure_cleanup();
        test_task_set_submission_failure_cleanup();
        test_zero_update_map();
        test_repeated_mapped_indices();
        for (bool one_column : {false, true})
            for (bool perturb : {false, true})
                test_distinct_copies(one_column, perturb);
        for (bool perturb : {false, true})
        {
            for (unsigned threads : {64u, 128u, 1024u})
                test_mapped(false, perturb, threads);
            for (unsigned threads : {64u, 256u, 1024u})
                test_mapped(true, perturb, threads);
        }
        test_invalid_factor(REAL(0), "zero diagonal");
        test_invalid_factor(std::numeric_limits<REAL>::quiet_NaN(), "NaN/Inf");
        std::cout << "All GPU numeric cases passed.\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "GPU kernel regression failed: " << error.what() << '\n';
        return 1;
    }
}
