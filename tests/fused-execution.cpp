#include "numeric.h"
#include "update-plan.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition,const char* message) {
    if(!condition) throw std::runtime_error(message);
}
template<class Error,class F> void rejects(F call,const char* message) {
    try { call(); } catch(const Error&) { return; }
    throw std::runtime_error(message);
}
std::unique_ptr<Symbolic_Matrix> matrix(unsigned n,const std::vector<unsigned>& columns,
        const std::vector<unsigned>& rows,const std::vector<double>& values) {
    auto a=std::make_unique<Symbolic_Matrix>(n,std::cout,std::cerr);
    // The legacy symbolic API takes mutable pointers but does not own them.
    auto ap=columns,ai=rows; auto ax=values;
    a->fill_in(ai.data(),ap.data()); a->csr(); a->predictLU(ai.data(),ap.data(),ax.data()); a->leveling();
    return a;
}
std::unique_ptr<Symbolic_Matrix> dense_matrix(unsigned n) {
    std::vector<unsigned> columns{0},rows;
    std::vector<double> values;
    for(unsigned c=0;c<n;++c) {
        for(unsigned r=0;r<n;++r) {
            rows.push_back(r);
            values.push_back(r==c ? 8.0+r%3 : double(int((r+2*c)%7)-3)/64);
        }
        columns.push_back(unsigned(rows.size()));
    }
    return matrix(n,columns,rows,values);
}
// Independent right-looking dense CPU LU. It does not read GPU update plans.
std::vector<REAL> cpu_factors(const Symbolic_Matrix& a,const REAL* input) {
    std::vector<long double> dense(size_t(a.n)*a.n,0);
    for(unsigned c=0;c<a.n;++c) for(unsigned p=a.sym_c_ptr[c];p<a.sym_c_ptr[c+1];++p)
        dense[size_t(c)*a.n+a.sym_r_idx[p]]=input[p];
    for(unsigned k=0;k<a.n;++k) {
        require(dense[size_t(k)*a.n+k]!=0,"CPU oracle has a zero pivot");
        for(unsigned r=k+1;r<a.n;++r) dense[size_t(k)*a.n+r]/=dense[size_t(k)*a.n+k];
        for(unsigned c=k+1;c<a.n;++c) for(unsigned r=k+1;r<a.n;++r)
            dense[size_t(c)*a.n+r]-=dense[size_t(k)*a.n+r]*dense[size_t(c)*a.n+k];
    }
    std::vector<REAL> factors(a.nnz);
    for(unsigned c=0;c<a.n;++c) for(unsigned p=a.sym_c_ptr[c];p<a.sym_c_ptr[c+1];++p)
        factors[p]=REAL(dense[size_t(c)*a.n+a.sym_r_idx[p]]);
    return factors;
}
void check(const Symbolic_Matrix& a,GpuTask& task,const std::vector<REAL>& input) {
    require(input.size()==size_t(a.nnz)*task.batch_size(),"incorrect test input size");
    task.check(); Symbolic_Matrix reference(a);
    const double tolerance=sizeof(REAL)==8 ? 2e-12 : 2e-5;
    for(unsigned copy=0;copy<task.batch_size();++copy) {
        const REAL* source=input.data()+size_t(copy)*a.nnz;
        const auto expected=cpu_factors(a,source); const REAL* actual=task.factors(copy);
        for(unsigned p=0;p<a.nnz;++p)
            require(std::isfinite(actual[p]) && std::abs(double(actual[p])-expected[p])<=tolerance*(1+std::abs(double(expected[p]))),
                    "Fused factors differ from independent CPU LU");
        reference.val.assign(source,source+a.nnz);
        const double residual=reference.factor_residual(actual);
        require(std::isfinite(residual) && residual<=tolerance,"Fused full factor residual failed");
    }
}
std::vector<REAL> repeated(const std::vector<REAL>& one,unsigned copies) {
    std::vector<REAL> result;
    for(unsigned c=0;c<copies;++c) result.insert(result.end(),one.begin(),one.end());
    return result;
}

void window_dependencies(unsigned copies,bool graph) {
    auto a=dense_matrix(40);
    const auto plan=make_panel_update_plan(*a,32,false,true);
    require(a->num_lev==40 && plan.deferred[31].end>plan.deferred[31].begin &&
            plan.immediate[0].end-plan.immediate[0].begin>256,"dense fixture lost window or update-loop coverage");
    GpuMatrix gpu(*a,1,false,UpdateMode::Fused,graph?SnapshotPolicy::Device:SnapshotPolicy::Shared);
    GpuTask task(gpu,copies);
    require(task.soa_value_stride()==0,"Fused unexpectedly uses SoA storage");
    if(graph) task.prepare_graph();
    task.enqueue(); task.wait(); check(*a,task,repeated(a->val,copies));
    for(unsigned round=1;round<=3;++round) {
        auto input=repeated(a->val,copies);
        for(unsigned copy=0;copy<copies;++copy) for(unsigned c=0;c<a->n;++c)
            for(unsigned p=a->sym_c_ptr[c];p<a->sym_c_ptr[c+1];++p) {
                const unsigned r=a->sym_r_idx[p];
                input[size_t(copy)*a->nnz+p]*=(REAL(1)+REAL((r+round+copy)%11)/64)*
                                             (REAL(1)+REAL((c+2*round+copy)%13)/64);
            }
        task.upload_input(input.data(),input.size()); task.enqueue(); task.wait(); check(*a,task,input);
    }
    task.enqueue(); task.wait(); check(*a,task,repeated(a->val,copies));
}

struct SparseFixture {
    std::unique_ptr<Symbolic_Matrix> a;
    std::vector<unsigned> positions,fill;
};
std::vector<REAL> sparse_input(const Symbolic_Matrix& a,unsigned round,unsigned copy) {
    std::vector<REAL> input(a.nnz,0);
    for(unsigned c=0;c<a.n;++c) for(unsigned p=a.sym_c_ptr[c];p<a.sym_c_ptr[c+1];++p) {
        const unsigned r=a.sym_r_idx[p],local_r=r%3,local_c=c%3;
        require(r/3==c/3,"block fixture is not block diagonal");
        if(r==c) input[p]=REAL(4)+REAL(round)/8+REAL(copy)/64+REAL(c/3)/512;
        else if(local_c==0) input[p]=local_r==1 ? REAL(.5)+REAL(copy)/64 : REAL(1);
        else if(local_c==1 && local_r==0) input[p]=round%2 ? REAL(0) : REAL(1);
        else if(local_c==2) input[p]=local_r==0 ? REAL(1)+REAL(round)/64 : (round%2 ? REAL(.0625) : REAL(0));
    }
    return input;
}
SparseFixture sparse_fixture() {
    constexpr unsigned n=33; // Eleven first-level columns, more than eight logical32 groups.
    std::vector<unsigned> columns{0},rows; std::vector<double> values;
    for(unsigned c=0;c<n;++c) {
        for(unsigned r=c/3*3;r<c/3*3+3;++r) if(!(r%3==2 && c%3==1)) { rows.push_back(r); values.push_back(0); }
        columns.push_back(unsigned(rows.size()));
    }
    SparseFixture fixture; fixture.a=matrix(n,columns,rows,values); auto& a=*fixture.a;
    a.val=sparse_input(a,0,0);
    for(unsigned c=0;c<n;++c) for(unsigned p=a.sym_c_ptr[c];p<a.sym_c_ptr[c+1];++p)
        (a.sym_r_idx[p]%3==2 && c%3==1 ? fixture.fill:fixture.positions).push_back(p);
    std::reverse(fixture.positions.begin(),fixture.positions.end()); // Compact order is caller-defined.
    require(a.nnz==99 && fixture.fill.size()==11 && a.level_ptr[1]==11,"sparse fixture lost fill or wide-level coverage");
    const auto lu=cpu_factors(a,a.val.data());
    for(unsigned p:fixture.fill) require(a.val[p]==0 && lu[p]!=0,"omitted input does not become nonzero fill");
    return fixture;
}
void compact_lifecycle(unsigned copies,bool graph) {
    auto fixture=sparse_fixture(); const auto& a=*fixture.a;
    std::unique_ptr<GpuTask> task;
    {
        GpuMatrix gpu(a,fixture.positions,1,false,UpdateMode::Fused);
        task=std::make_unique<GpuTask>(gpu,copies);
    } // Task must retain the Fused stage table, plan and compact position map.
    if(graph) { task->prepare_graph(); task->prepare_graph(); }
    for(unsigned round=0;round<3;++round) {
        std::vector<REAL> input,compact;
        for(unsigned copy=0;copy<copies;++copy) {
            const auto one=sparse_input(a,round,copy); input.insert(input.end(),one.begin(),one.end());
            for(unsigned p:fixture.positions) compact.push_back(one[p]);
        }
        task->upload_compact_input(compact.data(),compact.size());
        if(round==0) {
            rejects<std::logic_error>([&]{task->wait();},"upload-only wait accepted");
            rejects<std::logic_error>([&]{task->factors();},"upload-only factors accepted");
            rejects<std::logic_error>([&]{task->prepare_graph();},"Graph preparation with queued input accepted");
        }
        task->enqueue();
        if(round==0) rejects<std::logic_error>([&]{task->enqueue();},"second outstanding enqueue accepted");
        task->wait(); check(a,*task,input);
        if(round==0) {
            // Poison every omitted slot through the full API. The next compact
            // upload must clear it, as well as the factorization's own fill values.
            for(unsigned copy=0;copy<copies;++copy) for(unsigned p:fixture.fill)
                input[size_t(copy)*a.nnz+p]=REAL(.125)+REAL(copy)/1024;
            task->upload_input(input.data(),input.size()); task->enqueue(); task->wait(); check(a,*task,input);
        }
    }
    task->enqueue(); task->wait(); check(a,*task,repeated(a.val,copies));
}

void perturbation() {
    auto a=dense_matrix(3);
    std::fill(a->val.begin(),a->val.end(),REAL(0));
    a->val[1]=REAL(1e-5); a->val[4]=REAL(2);
    // First pivot has a nonzero L entry; the final zero pivot has no lower entries.
    // No Schur correction changes the diagonals, so the perturbed matrix is known.
    const REAL threshold=REAL(2)*REAL(3.45e-4f);
    auto modified=a->val; modified[0]=threshold; modified[8]=threshold;
    std::vector<unsigned> positions(a->nnz); std::iota(positions.begin(),positions.end(),0);
    GpuMatrix gpu(*a,positions,1,true,UpdateMode::Fused); GpuTask task(gpu,3); task.prepare_graph();
    const auto input=repeated(a->val,3);
    rejects<std::invalid_argument>([&]{task.upload_input(input.data(),input.size());},"perturbed fresh input accepted");
    rejects<std::invalid_argument>([&]{task.upload_compact_input(input.data(),input.size());},"perturbed compact input accepted");
    for(unsigned round=0;round<2;++round) { task.enqueue(); task.wait(); check(*a,task,repeated(modified,3)); }
}
} // namespace

int main() {
    try {
        window_dependencies(8,false); window_dependencies(33,true);
        compact_lifecycle(8,false); compact_lifecycle(33,true);
        perturbation();
        std::cout<<"PASS Fused window dependencies, AoS copies, compact lifecycle and perturbation\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<"Fused regression failed: "<<error.what()<<'\n'; return 1; }
}
