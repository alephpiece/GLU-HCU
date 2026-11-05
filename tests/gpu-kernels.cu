// Numeric kernel and host scheduling regressions, without preprocessing.
// Compile this translation unit instead of numeric.cu, and link symbolic.cc.
#include "../src/numeric.cu"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check_gpu(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename T> class DeviceBuffer {
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

struct Stream {
    cudaStream_t value;
    Stream() { check_gpu(cudaStreamCreate(&value), "create stream"); }
    ~Stream() { cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

struct Matrix {
    unsigned n;
    std::vector<unsigned> column, row, diagonal;
    std::vector<unsigned> csr_row, csr_column, csr_diagonal;
    std::vector<int> sources;
    std::vector<REAL> values;
};

// The two source columns are independent, but update common destinations.
// Neighbouring logical groups have unequal loop lengths on a 64-lane wave.
// There are enough destinations to reuse every group even in a 1024-thread
// block. Prefix lengths put pivots at lanes 0/31 and across chunk boundaries;
// long destinations also exercise the one-column kernel's 1024-element loop.
Matrix make_matrix(bool perturb)
{
    Matrix matrix;
    matrix.n = 1409;
    matrix.sources = {70, 73};
    std::vector<std::vector<std::pair<unsigned, REAL> > > columns(matrix.n);
    for (unsigned column = 0; column < matrix.n; ++column)
        columns[column].push_back(std::make_pair(column, REAL(4)));

    for (size_t i = 0; i < matrix.sources.size(); ++i) {
        const unsigned column = matrix.sources[i];
        columns[column][0].second = perturb ? (i ? REAL(-1e-12) : REAL(0)) : REAL(4 * (i + 1));
        for (unsigned row = 74; row < matrix.n; ++row) {
            if (row % 5 != 0)
                columns[column].push_back(std::make_pair(row, REAL(int(row % 9) - 4) / REAL(8)));
        }
    }

    const unsigned prefixes[] = {0, 1, 31, 32, 63, 64, 70};
    const unsigned tails[] = {0, 1, 7, 30, 31, 32, 33, 63, 65, 96, 1025, 1100};
    for (unsigned target = 0; target < 67; ++target) {
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
        for (unsigned row : rows) {
            const REAL value = REAL(0.5) + REAL(column % 11) / REAL(16) + REAL(row % 13) / REAL(32);
            columns[column].push_back(std::make_pair(row, value));
        }
    }

    matrix.column.push_back(0);
    matrix.diagonal.resize(matrix.n);
    std::vector<std::vector<unsigned> > rows(matrix.n);
    for (unsigned column = 0; column < matrix.n; ++column) {
        for (const auto& entry : columns[column]) {
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
    for (unsigned row = 0; row < matrix.n; ++row) {
        for (unsigned column : rows[row]) {
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
    for (int source : matrix.sources) {
        const unsigned diagonal = matrix.diagonal[source];
        if (perturb && std::fabs(expected[diagonal]) < threshold)
            expected[diagonal] = threshold;
        std::vector<REAL> lower(matrix.n, REAL(0));
        for (unsigned p = diagonal + 1; p < matrix.column[source + 1]; ++p) {
            expected[p] /= expected[diagonal];
            lower[matrix.row[p]] = expected[p];
        }
        for (unsigned q = matrix.csr_diagonal[source] + 1; q < matrix.csr_row[source + 1]; ++q) {
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

struct DeviceMatrix {
    DeviceBuffer<unsigned> column, row, diagonal, csr_row, csr_column, csr_diagonal;
    DeviceBuffer<int> sources;
    DeviceBuffer<REAL> values, temporary;

    explicit DeviceMatrix(const Matrix& matrix)
        : column(matrix.column), row(matrix.row), diagonal(matrix.diagonal),
          csr_row(matrix.csr_row), csr_column(matrix.csr_column), csr_diagonal(matrix.csr_diagonal),
          sources(matrix.sources), values(matrix.values), temporary(size_t(matrix.n) * matrix.sources.size())
    {
        check_gpu(cudaDeviceSynchronize(), "initialize test data");
    }
};

void compare(const std::vector<REAL>& actual, const std::vector<REAL>& expected, const char* field)
{
    if (actual.size() != expected.size())
        throw std::runtime_error(std::string(field) + " size mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        const REAL tolerance = REAL(64) * std::numeric_limits<REAL>::epsilon() *
                               std::max(REAL(1), std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance) {
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
        RL<UpdateMode::Scan, true><<<blocks, threads>>>(
            device.column.data(), device.row.data(), device.values.data(), device.diagonal.data(),
            device.csr_row.data(), device.csr_column.data(), device.csr_diagonal.data(), device.sources.data(),
            device.temporary.data(), matrix.n, 0, 0, 0, 0, threshold);
    else
        RL<UpdateMode::Scan><<<blocks, threads>>>(
            device.column.data(), device.row.data(), device.values.data(), device.diagonal.data(),
            device.csr_row.data(), device.csr_column.data(), device.csr_diagonal.data(), device.sources.data(),
            device.temporary.data(), matrix.n, 0, 0);
    check_gpu(cudaGetLastError(), "launch batched kernel");
    verify(device, matrix, perturb, threshold);
    std::cout << "PASS " << name << '\n';
}

void test_one_column(const Matrix& matrix, bool perturb, const std::string& name, unsigned threads = 1024)
{
    const REAL threshold = REAL(0.5);
    DeviceMatrix device(matrix);
    Stream streams[2];
    for (size_t slot = 0; slot < matrix.sources.size(); ++slot) {
        const unsigned source = matrix.sources[slot];
        if (perturb)
            RL_onecol_factorizeCurrentCol<true><<<1, threads, 0, streams[slot].value>>>(
                device.column.data(), device.row.data(), device.values.data(), device.diagonal.data(), source,
                device.temporary.data(), slot, matrix.n, 0, 0, threshold);
        else
            RL_onecol_factorizeCurrentCol<><<<1, threads, 0, streams[slot].value>>>(
                device.column.data(), device.row.data(), device.values.data(), device.diagonal.data(), source,
                device.temporary.data(), slot, matrix.n);
        check_gpu(cudaGetLastError(), "launch factor kernel");
        const unsigned targets = matrix.csr_row[source + 1] - matrix.csr_diagonal[source] - 1;
        if (targets) {
            RL_onecol_updateSubmat<UpdateMode::Scan><<<targets, threads, 0, streams[slot].value>>>(
                device.column.data(), device.row.data(), device.values.data(), device.csr_column.data(),
                device.csr_diagonal.data(), source, device.temporary.data(), slot, matrix.n);
            check_gpu(cudaGetLastError(), "launch update kernel");
        }
        RL_onecol_cleartmpMem<<<1, threads, 0, streams[slot].value>>>(
            device.column.data(), device.row.data(), device.diagonal.data(), source,
            device.temporary.data(), slot, matrix.n);
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


void verify_template_bytes(const GpuMatrix& gpu, const Symbolic_Matrix& matrix, UpdateMode mode)
{
    const size_t structure = sizeof(unsigned)*(matrix.sym_c_ptr.size()+matrix.l_col_ptr.size()+matrix.csr_r_ptr.size()+matrix.csr_diag_ptr.size()+
        (mode==UpdateMode::Scan ? matrix.sym_r_idx.size()+matrix.csr_c_idx.size() : 0))+
        sizeof(int)*matrix.level_idx.size();
    uint64_t updates = 0;
    for (unsigned c=0; c<matrix.n; ++c)
        updates += uint64_t(matrix.sym_c_ptr[c+1]-matrix.l_col_ptr[c]-1)*
                   (matrix.csr_r_ptr[c+1]-matrix.csr_diag_ptr[c]-1);
    const size_t map = mode == UpdateMode::Mapped
        ? sizeof(uint64_t)*(size_t(matrix.n)+1)+sizeof(unsigned)*(size_t(updates)+matrix.csr_c_idx.size()) : 0;
    const size_t task = sizeof(REAL)*(matrix.val.size()+
        (mode == UpdateMode::Scan ? size_t(matrix.n)*gpu.tmp_slots() : 0));
    if (gpu.task_device_bytes() != task)
        throw std::runtime_error("incorrect task device byte accounting");
    size_t extra=0;
    if(mode==UpdateMode::Mapped) {
        extra=sizeof(uint64_t)*(matrix.level_idx.size()+1);
        for(size_t lev=0;lev+1<matrix.level_ptr.size();++lev)
            if(matrix.level_ptr[lev+1]-matrix.level_ptr[lev]<=16)
                extra+=sizeof(int)*size_t(matrix.level_ptr[lev+1]-matrix.level_ptr[lev]);
    }
    for(unsigned copies : {1u,2u,3u,31u,32u,33u,42u}) {
        const unsigned lanes=1u<<filled_copy_bits(copies,5);
        const unsigned pitch=mode==UpdateMode::Mapped && copies>1 ? (copies+lanes-1)/lanes*lanes : 0;
        const size_t expected=sizeof(REAL)*(matrix.val.size()*(copies+pitch)+
            (mode==UpdateMode::Scan ? size_t(matrix.n)*gpu.tmp_slots()*copies : 0));
        if(gpu.task_device_bytes(copies)!=expected) throw std::runtime_error("incorrect batch device bytes");
    }
    if (gpu.update_map_bytes() != map || gpu.shared_device_bytes() != structure+map+extra)
        throw std::runtime_error("incorrect template device byte accounting");
    if (!std::isfinite(gpu.update_map_setup_ms()) || gpu.update_map_setup_ms() < 0 ||
        (mode == UpdateMode::Scan && gpu.update_map_setup_ms() != 0))
        throw std::runtime_error("incorrect map setup timing");
}

void test_repeated_tasks(bool graph = false, unsigned copies = 1, UpdateMode mode = UpdateMode::Scan)
{
    std::ostringstream out, err;
    std::vector<std::unique_ptr<GpuTask>> tasks;
    std::vector<std::vector<REAL>> expected;
    for (unsigned variant=0; variant<2; ++variant) {
        // 34 independent columns, then one dependent column. Exercises batched
        // chunking with fewer slots than the level width, then the small path.
        const unsigned n=35;
        const double diagonal=4+variant;
        std::vector<unsigned> col{0}, row;
        std::vector<double> val;
        std::vector<REAL> factors;
        for (unsigned j=0; j<n-1; ++j) {
            row.push_back(j); val.push_back(diagonal); factors.push_back(REAL(diagonal));
            row.push_back(n-1); val.push_back(1); factors.push_back(REAL(1/diagonal));
            col.push_back(unsigned(row.size()));
        }
        for (unsigned i=0; i<n; ++i) {
            row.push_back(i); val.push_back(i==n-1 ? 50 : 1);
            factors.push_back(i==n-1 ? REAL(50-(n-1)/diagonal) : REAL(1));
        }
        col.push_back(unsigned(row.size()));
        Symbolic_Matrix a(n,out,err);
        a.fill_in(row.data(),col.data()); a.csr();
        a.predictLU(row.data(),col.data(),val.data()); a.leveling();
        GpuMatrix matrix(a, variant ? 3 : 1, false, mode);
        verify_template_bytes(matrix, a, mode);
        const unsigned expected_slots = mode == UpdateMode::Scan ? (variant ? 3u : 1u) : 0;
        if (matrix.tmp_slots() != expected_slots) throw std::runtime_error("wrong effective scratch slots");
        for (unsigned copy=0; copy<2; ++copy) {
            tasks.emplace_back(new GpuTask(matrix, copies));
            if (graph) { tasks.back()->prepare_graph(); tasks.back()->prepare_graph(); }
            expected.push_back(factors);
        }
        // Both the host matrix and GpuMatrix wrapper die here. Tasks must retain
        // all input, structure and launch metadata required by later enqueues.
    }
    for (unsigned loop=0; loop<4; ++loop) {
        if(loop%2) for(auto& task : tasks) {
            task->upload_input();
            bool rejected_read=false,rejected_requeue=false,rejected_capture=false,rejected_wait=false;
            try { task->factors(); } catch(const std::logic_error&) { rejected_read=true; }
            try { task->upload_input(); } catch(const std::logic_error&) { rejected_requeue=true; }
            try { task->prepare_graph(); } catch(const std::logic_error&) { rejected_capture=true; }
            try { task->wait(); } catch(const std::logic_error&) { rejected_wait=true; }
            if(!rejected_read || !rejected_requeue || !rejected_capture || !rejected_wait)
                throw std::runtime_error("queued input allowed invalid reuse or result access");
        }
        for (auto& task : tasks) task->enqueue();
        if (loop == 0) {
            bool rejected_read=false, rejected_reuse=false, rejected_capture=false;
            try { tasks[0]->factors(); } catch (const std::logic_error&) { rejected_read=true; }
            try { tasks[0]->enqueue(); } catch (const std::logic_error&) { rejected_reuse=true; }
            try { tasks[0]->prepare_graph(); } catch (const std::logic_error&) { rejected_capture=true; }
            if (!rejected_read || !rejected_reuse || !rejected_capture)
                throw std::runtime_error("pending task allowed read or reuse");
        }
        for (auto& task : tasks) task->wait();
        for (size_t i=0; i<tasks.size(); ++i) {
            tasks[i]->check();
            for (unsigned copy=0; copy<copies; ++copy)
                compare(std::vector<REAL>(tasks[i]->factors(copy),tasks[i]->factors(copy)+expected[i].size()),
                    expected[i], "repeated independent tasks");
        }
    }
    std::cout << "PASS host/repeated-mixed-tasks-and-template-lifetime/"
              << (mode == UpdateMode::Scan ? "scan" : "mapped") << '\n';
}

void test_zero_update_map()
{
    unsigned column[]{0, 1, 2}, row[]{0, 1};
    double values[]{2, 4};
    std::ostringstream out, err;
    Symbolic_Matrix matrix(2, out, err);
    matrix.fill_in(row, column); matrix.csr();
    matrix.predictLU(row, column, values); matrix.leveling();
    GpuMatrix gpu(matrix, 1, false, UpdateMode::Mapped);
    verify_template_bytes(gpu, matrix, UpdateMode::Mapped);
    GpuTask task(gpu);
    task.prepare_graph(); task.enqueue(); task.wait(); task.check();
    compare(std::vector<REAL>(task.factors(), task.factors()+2), {REAL(2),REAL(4)}, "empty update map");
    std::cout << "PASS host/mapped-zero-update-entries\n";
}

void test_distinct_copies(bool one_column, bool perturb)
{
    const unsigned n=2, copies=3, slots=2, nnz=4;
    DeviceBuffer<unsigned> col(std::vector<unsigned>{0,2,4}), row(std::vector<unsigned>{0,1,0,1});
    DeviceBuffer<unsigned> diag(std::vector<unsigned>{0,3}), csr_row(std::vector<unsigned>{0,2,4});
    DeviceBuffer<unsigned> csr_col(std::vector<unsigned>{0,1,0,1}), csr_diag(std::vector<unsigned>{0,3});
    DeviceBuffer<int> levels(std::vector<int>{0});
    std::vector<REAL> input, expected;
    for (unsigned i=0; i<copies; ++i) {
        const REAL a=perturb ? REAL(0) : REAL(i+2), b=REAL(i+5), d=REAL(i+10);
        const REAL pivot=perturb ? REAL(2) : a;
        input.insert(input.end(),{a,REAL(i+1),b,d});
        expected.insert(expected.end(),{pivot,REAL(i+1)/pivot,b,d-REAL(i+1)/pivot*b});
    }
    DeviceBuffer<REAL> values(input), scratch(n*slots*copies);
    Stream stream;
    if (one_column) {
        if (perturb)
            RL_onecol_factorizeCurrentCol<true><<<dim3(1,copies),1024,0,stream.value>>>(
                col.data(), row.data(), values.data(), diag.data(), 0, scratch.data(), 1, n, nnz, n*slots,
                REAL(2));
        else
            RL_onecol_factorizeCurrentCol<><<<dim3(1,copies),1024,0,stream.value>>>(
                col.data(), row.data(), values.data(), diag.data(), 0, scratch.data(), 1, n, nnz, n*slots);
        RL_onecol_updateSubmat<UpdateMode::Scan><<<dim3(1,copies),1024,0,stream.value>>>(
            col.data(), row.data(), values.data(), csr_col.data(), csr_diag.data(), 0, scratch.data(), 1, n, nnz,
            n*slots);
        RL_onecol_cleartmpMem<<<dim3(1,copies),1024,0,stream.value>>>(
            col.data(),row.data(),diag.data(),0,scratch.data(),1,n,n*slots);
    } else if (perturb)
        RL<UpdateMode::Scan, true><<<dim3(1,copies),64,0,stream.value>>>(
            col.data(), row.data(), values.data(), diag.data(), csr_row.data(), csr_col.data(), csr_diag.data(),
            levels.data(), scratch.data(), n, 0, 0, nnz, n*slots, REAL(2));
    else
        RL<UpdateMode::Scan><<<dim3(1,copies),64,0,stream.value>>>(
            col.data(), row.data(), values.data(), diag.data(), csr_row.data(), csr_col.data(), csr_diag.data(),
            levels.data(), scratch.data(), n, 0, 0, nnz, n*slots);
    check_gpu(cudaGetLastError(), "launch distinct-copy kernels");
    check_gpu(cudaStreamSynchronize(stream.value), "wait distinct copies");
    compare(values.read(),expected,"distinct batch values");
    compare(scratch.read(),std::vector<REAL>(n*slots*copies,REAL(0)),"distinct batch scratch cleanup");
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
    for (unsigned c=0; c<matrix.n; ++c)
        columns[c].emplace_back(c, REAL(4));
    for (unsigned s=0; s<matrix.sources.size(); ++s) {
        const unsigned c = matrix.sources[s];
        columns[c][0].second = perturb ? (s % 2 ? REAL(-1e-12) : REAL(0))
                                      : REAL(2u << (s+copy));
        for (unsigned i=0; i<lengths[s]; ++i)
            columns[c].emplace_back(20+i, REAL(int(i % 7)-3)*REAL(copy+1)/REAL(8));
    }
    for (unsigned t=0; t<5; ++t) {
        const unsigned c = 90+t;
        std::vector<unsigned> rows{0, 6, 87, c};
        for (unsigned s=0; s<matrix.sources.size(); ++s) {
            if (t >= targets[s]) continue;
            rows.push_back(matrix.sources[s]);
            for (unsigned i=0; i<lengths[s]; ++i) rows.push_back(20+i);
        }
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        columns[c].clear();
        for (unsigned row : rows)
            columns[c].emplace_back(row, REAL(0.5)+REAL(row % 7)/REAL(16)+
                                         REAL(t)/REAL(32)+REAL(copy)/REAL(8));
    }
    matrix.column.push_back(0);
    matrix.diagonal.resize(matrix.n);
    std::vector<std::vector<unsigned>> rows(matrix.n);
    for (unsigned c=0; c<matrix.n; ++c) {
        for (const auto& entry : columns[c]) {
            if (entry.first == c) matrix.diagonal[c] = matrix.row.size();
            matrix.row.push_back(entry.first);
            matrix.values.push_back(entry.second);
            rows[entry.first].push_back(c);
        }
        matrix.column.push_back(matrix.row.size());
    }
    matrix.csr_row.push_back(0);
    matrix.csr_diagonal.resize(matrix.n);
    for (unsigned row=0; row<matrix.n; ++row) {
        for (unsigned c : rows[row]) {
            if (c == row) matrix.csr_diagonal[row] = matrix.csr_column.size();
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
    const size_t nnz = matrix.values.size(), stride = size_t(matrix.n)*slots;
    // Use a dense coordinate lookup as an independent map oracle, rather than
    // duplicating the production builder's sorted-column merge.
    const unsigned missing = std::numeric_limits<unsigned>::max();
    std::vector<unsigned> positions(size_t(matrix.n)*matrix.n, missing);
    for (unsigned c=0; c<matrix.n; ++c)
        for (unsigned p=matrix.column[c]; p<matrix.column[c+1]; ++p)
            positions[size_t(c)*matrix.n+matrix.row[p]] = p;
    std::vector<uint64_t> bases(matrix.n+1);
    std::vector<unsigned> destinations, pivots(matrix.csr_column.size(), missing);
    for (unsigned c=0; c<matrix.n; ++c) {
        bases[c] = destinations.size();
        for (unsigned q=matrix.csr_diagonal[c]+1; q<matrix.csr_row[c+1]; ++q) {
            const unsigned target = matrix.csr_column[q];
            pivots[q] = positions[size_t(target)*matrix.n+c];
            if (pivots[q] == missing) throw std::runtime_error("mapped fixture missing U entry");
            for (unsigned p=matrix.diagonal[c]+1; p<matrix.column[c+1]; ++p) {
                const unsigned destination = positions[size_t(target)*matrix.n+matrix.row[p]];
                if (destination == missing) throw std::runtime_error("mapped fixture is not closed");
                destinations.push_back(destination);
            }
        }
    }
    bases[matrix.n] = destinations.size();
    std::vector<REAL> input, expected;
    for (unsigned copy=0; copy<copies; ++copy) {
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
    for (unsigned repeat=0; repeat<2; ++repeat) {
        // No scratch allocation: repeated mapped execution only restores values.
        check_gpu(cudaMemcpy(values.data(), input.data(), input.size()*sizeof(REAL),
                             cudaMemcpyHostToDevice), "restore mapped inputs");
        if (one_column) {
            for (int c : matrix.sources) {
                if (perturb)
                    RL_onecol_factorizeCurrentCol<true, false><<<dim3(1,copies),threads,0,stream.value>>>(
                        column.data(), row.data(), values.data(), diagonal.data(), c, nullptr, 0,
                        matrix.n, nnz, stride, threshold);
                else
                    RL_onecol_factorizeCurrentCol<false, false><<<dim3(1,copies),threads,0,stream.value>>>(
                        column.data(), row.data(), values.data(), diagonal.data(), c, nullptr, 0,
                        matrix.n, nnz, stride);
                check_gpu(cudaGetLastError(), "launch mapped source normalization");
                const unsigned targets = matrix.csr_row[c+1]-matrix.csr_diagonal[c]-1;
                RL_onecol_updateSubmat<UpdateMode::Mapped><<<dim3(targets,copies),threads,0,stream.value>>>(
                    column.data(), row.data(), values.data(), csr_column.data(), csr_diagonal.data(), c,
                    nullptr, 0, matrix.n, nnz, stride, diagonal.data(), map_bases.data(),
                    map_destinations.data(), map_pivots.data());
                check_gpu(cudaGetLastError(), "launch mapped column update");
            }
        } else {
            for (unsigned offset : {0u, 2u}) {
                if (perturb)
                    RL<UpdateMode::Mapped, true><<<dim3(slots,copies),threads,0,stream.value>>>(
                        column.data(), row.data(), values.data(), diagonal.data(), csr_row.data(),
                        csr_column.data(), csr_diagonal.data(), sources.data(), nullptr, matrix.n, 1,
                        offset, nnz, stride, threshold, map_bases.data(), map_destinations.data(),
                        map_pivots.data());
                else
                    RL<UpdateMode::Mapped><<<dim3(slots,copies),threads,0,stream.value>>>(
                        column.data(), row.data(), values.data(), diagonal.data(), csr_row.data(),
                        csr_column.data(), csr_diagonal.data(), sources.data(), nullptr, matrix.n, 1,
                        offset, nnz, stride, 0, map_bases.data(), map_destinations.data(), map_pivots.data());
                check_gpu(cudaGetLastError(), "launch mapped level chunk");
            }
        }
        check_gpu(cudaStreamSynchronize(stream.value), "wait mapped kernels");
        compare(values.read(), expected, "mapped factors and distinct copies");
    }
    std::cout << "PASS mapped/" << (one_column ? "one-column/" : "RL/")
              << (perturb ? "perturb/" : "plain/") << threads << '\n';
}

Matrix make_layout_matrix(bool perturb, unsigned copy,unsigned variant = 0)
{
    Matrix matrix;
    matrix.n = 100;
    matrix.sources = {2, 5, 8, 11};
    const unsigned shapes[][4]={{65,33,1,0},{0,1,2,3},{17,31,32,33},{9,3,2,0},{7,8,9,0}};
    const unsigned fanouts[][4]={{5,3,4,2},{5,1,0,2},{5,3,4,2},{5,5,5,2},{5,3,0,2}};
    if(variant>=5) throw std::invalid_argument("unknown mapped fixture");
    const unsigned* lengths=shapes[variant];
    const unsigned* targets=fanouts[variant];
    std::vector<std::vector<std::pair<unsigned, REAL>>> columns(matrix.n);
    for (unsigned c=0; c<matrix.n; ++c)
        columns[c].emplace_back(c, REAL(4));
    for (unsigned s=0; s<matrix.sources.size(); ++s) {
        const unsigned c = matrix.sources[s];
        columns[c][0].second = perturb ? (s % 2 ? REAL(-1e-12) : REAL(0))
                                      : REAL(2u << s)*(REAL(1)+REAL(copy)/REAL(4));
        for (unsigned i=0; i<lengths[s]; ++i)
            columns[c].emplace_back(20+i, REAL(int(i % 7)-3)*REAL(copy+1)/REAL(8));
    }
    for (unsigned t=0; t<5; ++t) {
        const unsigned c = 90+t;
        std::vector<unsigned> rows{0, 6, 87, c};
        for (unsigned s=0; s<matrix.sources.size(); ++s) {
            if (t >= targets[s]) continue;
            rows.push_back(matrix.sources[s]);
            for (unsigned i=0; i<lengths[s]; ++i) rows.push_back(20+i);
        }
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        columns[c].clear();
        for (unsigned row : rows)
            columns[c].emplace_back(row, REAL(0.5)+REAL(row % 7)/REAL(16)+
                                         REAL(t)/REAL(32)+REAL(copy)/REAL(8));
    }
    matrix.column.push_back(0);
    matrix.diagonal.resize(matrix.n);
    std::vector<std::vector<unsigned>> rows(matrix.n);
    for (unsigned c=0; c<matrix.n; ++c) {
        for (const auto& entry : columns[c]) {
            if (entry.first == c) matrix.diagonal[c] = matrix.row.size();
            matrix.row.push_back(entry.first);
            matrix.values.push_back(entry.second);
            rows[entry.first].push_back(c);
        }
        matrix.column.push_back(matrix.row.size());
    }
    matrix.csr_row.push_back(0);
    matrix.csr_diagonal.resize(matrix.n);
    for (unsigned row=0; row<matrix.n; ++row) {
        for (unsigned c : rows[row]) {
            if (c == row) matrix.csr_diagonal[row] = matrix.csr_column.size();
            matrix.csr_column.push_back(c);
        }
        matrix.csr_row.push_back(matrix.csr_column.size());
    }
    return matrix;
}

void test_layout(unsigned copies,unsigned variant,bool perturb,unsigned path) {
    const bool soa=path!=0,wide=path==3,normalized=path==2;
    const unsigned bits=filled_copy_bits(copies,wide ? 3 : 5),lanes=1u<<bits;
    const unsigned alignment=1u<<filled_copy_bits(copies,5);
    const unsigned pitch=(copies+alignment-1)/alignment*alignment,tiles_y=(copies+lanes-1)/lanes;
    const REAL threshold=REAL(0.5);
    const Matrix matrix=make_layout_matrix(perturb,0,variant);
    const size_t nnz=matrix.values.size();
    // Use a dense coordinate lookup as an independent map oracle, rather than
    // duplicating the production builder's sorted-column merge.
    const unsigned missing = std::numeric_limits<unsigned>::max();
    std::vector<unsigned> positions(size_t(matrix.n)*matrix.n, missing);
    for (unsigned c=0; c<matrix.n; ++c)
        for (unsigned p=matrix.column[c]; p<matrix.column[c+1]; ++p)
            positions[size_t(c)*matrix.n+matrix.row[p]] = p;
    std::vector<uint64_t> bases(matrix.n+1);
    std::vector<unsigned> destinations, pivots(matrix.csr_column.size(), missing);
    for (unsigned c=0; c<matrix.n; ++c) {
        bases[c] = destinations.size();
        for (unsigned q=matrix.csr_diagonal[c]+1; q<matrix.csr_row[c+1]; ++q) {
            const unsigned target = matrix.csr_column[q];
            pivots[q] = positions[size_t(target)*matrix.n+c];
            if (pivots[q] == missing) throw std::runtime_error("mapped fixture missing U entry");
            for (unsigned p=matrix.diagonal[c]+1; p<matrix.column[c+1]; ++p) {
                const unsigned destination = positions[size_t(target)*matrix.n+matrix.row[p]];
                if (destination == missing) throw std::runtime_error("mapped fixture is not closed");
                destinations.push_back(destination);
            }
        }
        std::vector<unsigned> unique(destinations.begin()+bases[c],destinations.end());
        std::sort(unique.begin(),unique.end());
        if (std::adjacent_find(unique.begin(),unique.end())!=unique.end())
            throw std::runtime_error("one mapped source has duplicate update destinations");
    }
    bases[matrix.n] = destinations.size();
    std::vector<REAL> input, expected;
    for (unsigned copy=0; copy<copies; ++copy) {
        const Matrix distinct = make_layout_matrix(perturb, copy,variant);
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
    std::vector<uint64_t> target_prefix{0,7};
    for (unsigned c : {2u,5u,8u,11u})
        target_prefix.push_back(target_prefix.back()+(matrix.column[c+1]>matrix.diagonal[c]+1
            ? matrix.csr_row[c+1]-matrix.csr_diagonal[c]-1 : 0));
    DeviceBuffer<uint64_t> prefix(target_prefix);
    DeviceBuffer<REAL> values(input),interleaved(soa ? nnz*pitch : 1);
    Stream stream;
    for(unsigned repeat=0;repeat<2;++repeat) {
        const REAL guard=repeat ? std::numeric_limits<REAL>::quiet_NaN() : REAL(12345.5);
        const std::vector<REAL> guards(soa ? nnz*pitch : 1,guard);
        check_gpu(cudaMemcpy(interleaved.data(),guards.data(),guards.size()*sizeof(REAL),cudaMemcpyHostToDevice),"initialize guards");
        check_gpu(cudaMemcpy(values.data(),input.data(),input.size()*sizeof(REAL),cudaMemcpyHostToDevice),"restore oracle inputs");
        const unsigned tiles=unsigned((nnz/32+(nnz%32!=0))*(copies/32+(copies%32!=0)));
        if(soa) transpose_task_values<<<tiles,dim3(32,8),0,stream.value>>>(values.data(),interleaved.data(),copies,unsigned(nnz),unsigned(nnz),pitch);
        if(wide) {
            if(perturb)
                mapped_wide_soa<true><<<dim3(4,tiles_y),256,0,stream.value>>>(interleaved.data(),copies,column.data(),diagonal.data(),
                    csr_row.data(),csr_diagonal.data(),sources.data(),1,map_bases.data(),map_destinations.data(),map_pivots.data(),threshold,bits,pitch);
            else
                mapped_wide_soa<false><<<dim3(4,tiles_y),256,0,stream.value>>>(interleaved.data(),copies,column.data(),diagonal.data(),
                    csr_row.data(),csr_diagonal.data(),sources.data(),1,map_bases.data(),map_destinations.data(),map_pivots.data(),0,bits,pitch);
        } else {
            if(normalized) {
                if(perturb) mapped_normalize_level_soa<true><<<dim3(4,tiles_y),256,0,stream.value>>>(interleaved.data(),copies,column.data(),diagonal.data(),sources.data(),1,bits,threshold,pitch);
                else mapped_normalize_level_soa<false><<<dim3(4,tiles_y),256,0,stream.value>>>(interleaved.data(),copies,column.data(),diagonal.data(),sources.data(),1,bits,0,pitch);
            }
            const unsigned targets=unsigned(target_prefix.back()-7);
            if(soa) {
                if(normalized) mapped_level_update_soa<false,true><<<dim3(targets,tiles_y),256,0,stream.value>>>(interleaved.data(),copies,column.data(),diagonal.data(),csr_diagonal.data(),sources.data(),prefix.data(),1,5,map_bases.data(),map_destinations.data(),map_pivots.data(),0,pitch,bits);
                else if(perturb) mapped_level_update_soa<true><<<dim3(targets,tiles_y),256,0,stream.value>>>(interleaved.data(),copies,column.data(),diagonal.data(),csr_diagonal.data(),sources.data(),prefix.data(),1,5,map_bases.data(),map_destinations.data(),map_pivots.data(),threshold,pitch,bits);
                else mapped_level_update_soa<false><<<dim3(targets,tiles_y),256,0,stream.value>>>(interleaved.data(),copies,column.data(),diagonal.data(),csr_diagonal.data(),sources.data(),prefix.data(),1,5,map_bases.data(),map_destinations.data(),map_pivots.data(),0,pitch,bits);
            } else {
                if(perturb) mapped_level_update<true><<<dim3(targets,copies),256,0,stream.value>>>(values.data(),nnz,column.data(),diagonal.data(),csr_diagonal.data(),sources.data(),prefix.data(),1,5,map_bases.data(),map_destinations.data(),map_pivots.data(),threshold);
                else mapped_level_update<false><<<dim3(targets,copies),256,0,stream.value>>>(values.data(),nnz,column.data(),diagonal.data(),csr_diagonal.data(),sources.data(),prefix.data(),1,5,map_bases.data(),map_destinations.data(),map_pivots.data(),0);
            }
        }
        if(soa) transpose_task_values<<<tiles,dim3(32,8),0,stream.value>>>(interleaved.data(),values.data(),unsigned(nnz),copies,pitch,unsigned(nnz));
        if(!wide && !normalized) {
            if(perturb) RL_onecol_factorizeCurrentCol<true,false,true><<<dim3(4,copies),256,0,stream.value>>>(column.data(),row.data(),values.data(),diagonal.data(),0,nullptr,0,matrix.n,nnz,0,threshold,sources.data(),1);
            else RL_onecol_factorizeCurrentCol<false,false,true><<<dim3(4,copies),256,0,stream.value>>>(column.data(),row.data(),values.data(),diagonal.data(),0,nullptr,0,matrix.n,nnz,0,0,sources.data(),1);
        }
        check_gpu(cudaGetLastError(),"launch layout oracle kernels");
        check_gpu(cudaStreamSynchronize(stream.value),"wait layout oracle");
        compare(values.read(),expected,"layout factors and distinct values");
        if(soa) {
            const auto actual=interleaved.read();
            for(size_t row=0;row<nnz;++row) for(unsigned copy=copies;copy<pitch;++copy)
                if(repeat ? !std::isnan(actual[row*pitch+copy]) : actual[row*pitch+copy]!=guard)
                    throw std::runtime_error("padding guard overwritten");
        }
    }
}

void test_small_level(bool graph,unsigned copies,bool perturb) {
    // First level has four sources: zero targets, two targets, empty L, one
    // target. The repeated prefixes are {0,0,2,2,3}; sources 1/3 overlap in col5.
    unsigned columns[]{0,2,5,6,8,10,15};
    unsigned rows[]{0,5, 1,4,5, 2, 3,5, 1,4, 1,2,3,4,5};
    double values[]{4,2, 8,1,2, 16, 32,2, 2,20, 1,4,2,1,30};
    if (perturb) values[5]=0;
    std::ostringstream out,err;
    Symbolic_Matrix matrix(6,out,err);
    matrix.fill_in(rows,columns); matrix.csr();
    matrix.predictLU(rows,columns,values); matrix.leveling();
    if (matrix.level_ptr[1]!=4) throw std::runtime_error("small-level fixture width changed");
    GpuMatrix control(matrix,1,perturb,UpdateMode::Scan);
    GpuTask reference(control,copies);
    reference.enqueue(); reference.wait(); reference.check();
    GpuMatrix candidate(matrix,1,perturb,UpdateMode::Mapped);
    verify_template_bytes(candidate,matrix,UpdateMode::Mapped);
    GpuTask task(candidate,copies);
    if (graph) task.prepare_graph();
    for (unsigned loop=0; loop<3; ++loop) {
        task.enqueue(); task.wait(); task.check();
        for (unsigned copy=0; copy<copies; ++copy) {
            compare(std::vector<REAL>(task.factors(copy),task.factors(copy)+matrix.nnz),
                std::vector<REAL>(reference.factors(copy),reference.factors(copy)+matrix.nnz),
                "two-stage unequal targets and empty L");
            if (!perturb && matrix.factor_residual(task.factors(copy))>(sizeof(REAL)==4 ? 1e-5 : 1e-12))
                throw std::runtime_error("two-stage full factor residual");
        }
    }
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

int main()
{
    try {
        GpuMatrix::select_device();
        cudaDeviceProp properties;
        check_gpu(cudaGetDeviceProperties(&properties, 0), "inspect device");
        std::cout << "Device: " << properties.name << ", warp size: " << properties.warpSize
                  << ", REAL bits: " << sizeof(REAL) * 8 << '\n';
        for (bool perturb : {false, true}) {
            const Matrix matrix = make_matrix(perturb);
            for (unsigned threads : {64u, 128u, 1024u})
                test_batched(matrix, threads, perturb,
                    std::string(perturb ? "RL_perturb/" : "RL/") + std::to_string(threads));
            for (unsigned threads : {256u, 1024u})
                test_one_column(matrix, perturb,
                    std::string(perturb ? "one-column/perturb/" : "one-column/plain/") + std::to_string(threads), threads);
        }
        const Matrix empty_lower = make_empty_lower_matrix();
        test_batched(empty_lower, 64, true, "RL_perturb/empty-lower");
        test_one_column(empty_lower, true, "one-column/empty-lower");
        test_limited_slots();
        for(UpdateMode mode : {UpdateMode::Scan,UpdateMode::Mapped})
            for(bool graph : {false,true})
                for(unsigned copies : {1u,2u,3u,16u,33u,42u})
                    test_repeated_tasks(graph,copies,mode);
        for(bool graph : {false,true})
            for(unsigned copies : {1u,2u,3u,33u,42u})
                for(bool perturb : {false,true}) test_small_level(graph,copies,perturb);
        for(unsigned copies : {1u,2u,3u,7u,8u,9u,31u,32u,33u,42u})
            for(unsigned variant=0;variant<5;++variant)
                for(bool perturb : {false,true})
                    for(unsigned path=0;path<4;++path) test_layout(copies,variant,perturb,path);
        test_zero_update_map();
        for (bool one_column : {false,true})
            for (bool perturb : {false,true}) test_distinct_copies(one_column,perturb);
        for (bool perturb : {false,true}) {
            for (unsigned threads : {64u, 128u, 1024u}) test_mapped(false,perturb,threads);
            for (unsigned threads : {64u, 256u, 1024u}) test_mapped(true,perturb,threads);
        }
        test_invalid_factor(REAL(0), "zero diagonal");
        test_invalid_factor(std::numeric_limits<REAL>::quiet_NaN(), "NaN/Inf");
        std::cout << "All GPU numeric cases passed.\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "GPU kernel regression failed: " << error.what() << '\n';
        return 1;
    }
}
