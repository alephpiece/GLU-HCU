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
        RL_perturb<<<blocks, threads>>>(device.column.data(), device.row.data(), device.values.data(),
            device.diagonal.data(), device.csr_row.data(), device.csr_column.data(), device.csr_diagonal.data(),
            device.sources.data(), device.temporary.data(), matrix.n, 0, 0, threshold);
    else
        RL<<<blocks, threads>>>(device.column.data(), device.row.data(), device.values.data(),
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
    for (size_t slot = 0; slot < matrix.sources.size(); ++slot) {
        const unsigned source = matrix.sources[slot];
        if (perturb)
            RL_onecol_factorizeCurrentCol_perturb<<<1, threads, 0, streams[slot].value>>>(
                device.column.data(), device.row.data(), device.values.data(), device.diagonal.data(),
                source, device.temporary.data(), slot, matrix.n, threshold);
        else
            RL_onecol_factorizeCurrentCol<<<1, threads, 0, streams[slot].value>>>(
                device.column.data(), device.row.data(), device.values.data(), device.diagonal.data(),
                source, device.temporary.data(), slot, matrix.n);
        check_gpu(cudaGetLastError(), "launch factor kernel");
        const unsigned targets = matrix.csr_row[source + 1] - matrix.csr_diagonal[source] - 1;
        if (targets) {
            RL_onecol_updateSubmat<<<targets, threads, 0, streams[slot].value>>>(
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


void test_repeated_tasks(bool graph = false, unsigned copies = 1)
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
        GpuMatrix matrix(a, variant ? 3 : 1, false);
        if (matrix.tmp_slots() != (variant ? 3u : 1u)) throw std::runtime_error("wrong scratch cap");
        for (unsigned copy=0; copy<2; ++copy) {
            tasks.emplace_back(new GpuTask(matrix, copies));
            if (graph) { tasks.back()->prepare_graph(); tasks.back()->prepare_graph(); }
            expected.push_back(factors);
        }
        // Both the host matrix and GpuMatrix wrapper die here. Tasks must retain
        // all input, structure and launch metadata required by later enqueues.
    }
    for (unsigned loop=0; loop<4; ++loop) {
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
    std::cout << "PASS host/repeated-mixed-tasks-and-template-lifetime\n";
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
            RL_onecol_factorizeCurrentCol_perturb<<<dim3(1,copies),1024,0,stream.value>>>(
                col.data(),row.data(),values.data(),diag.data(),0,scratch.data(),1,n,REAL(2),nnz,n*slots);
        else
            RL_onecol_factorizeCurrentCol<<<dim3(1,copies),1024,0,stream.value>>>(
                col.data(),row.data(),values.data(),diag.data(),0,scratch.data(),1,n,nnz,n*slots);
        RL_onecol_updateSubmat<<<dim3(1,copies),1024,0,stream.value>>>(
            col.data(),row.data(),values.data(),csr_col.data(),csr_diag.data(),0,scratch.data(),1,n,nnz,n*slots);
        RL_onecol_cleartmpMem<<<dim3(1,copies),1024,0,stream.value>>>(
            col.data(),row.data(),diag.data(),0,scratch.data(),1,n,n*slots);
    } else if (perturb)
        RL_perturb<<<dim3(1,copies),64,0,stream.value>>>(col.data(),row.data(),values.data(),diag.data(),
            csr_row.data(),csr_col.data(),csr_diag.data(),levels.data(),scratch.data(),n,0,0,REAL(2),nnz,n*slots);
    else
        RL<<<dim3(1,copies),64,0,stream.value>>>(col.data(),row.data(),values.data(),diag.data(),
            csr_row.data(),csr_col.data(),csr_diag.data(),levels.data(),scratch.data(),n,0,0,nnz,n*slots);
    check_gpu(cudaGetLastError(), "launch distinct-copy kernels");
    check_gpu(cudaStreamSynchronize(stream.value), "wait distinct copies");
    compare(values.read(),expected,"distinct batch values");
    compare(scratch.read(),std::vector<REAL>(n*slots*copies,REAL(0)),"distinct batch scratch cleanup");
    std::cout << "PASS distinct batch copies " << one_column << ' ' << perturb << '\n';
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
        check_gpu(cudaSetDevice(0), "select device");
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
        test_repeated_tasks();
        test_repeated_tasks(true);
        test_repeated_tasks(false,3);
        test_repeated_tasks(true,3);
        test_repeated_tasks(false,16);
        test_repeated_tasks(true,16);
        for (bool one_column : {false,true})
            for (bool perturb : {false,true}) test_distinct_copies(one_column,perturb);
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
