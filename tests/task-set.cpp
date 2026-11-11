#include "numeric.h"
#include "update-plan.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
template<class Error,class F> void rejects(F call,const char* message) {
    try {call();}catch(const Error&){return;}
    throw std::runtime_error(message);
}
struct Fixture {
    std::unique_ptr<Symbolic_Matrix> a;
    std::vector<unsigned> positions,fill;
};
Fixture fixture(unsigned kind) {
    const unsigned n=kind==0?40:kind==1?33:kind==2?5:160;
    std::vector<unsigned> columns{0},rows;
    std::vector<double> values;
    for(unsigned c=0;c<n;++c) {
        for(unsigned r=0;r<n;++r) {
            if(kind==1 && (r/3!=c/3 || (r%3==2 && c%3==1)))continue;
            if(kind==2 && r!=c)continue;
            rows.push_back(r);
            values.push_back(r==c?8.0+c%3:double(1+(r+2*c)%7)/64);
        }
        columns.push_back(unsigned(rows.size()));
    }
    Fixture f; f.a=std::make_unique<Symbolic_Matrix>(n,std::cout,std::cerr);
    f.a->fill_in(rows.data(),columns.data()); f.a->csr();
    f.a->predictLU(rows.data(),columns.data(),values.data()); f.a->leveling();
    for(unsigned c=0;c<n;++c)for(unsigned p=f.a->sym_c_ptr[c];p<f.a->sym_c_ptr[c+1];++p)
        (kind==1 && f.a->sym_r_idx[p]%3==2 && c%3==1?f.fill:f.positions).push_back(p);
    std::reverse(f.positions.begin(),f.positions.end());
    require(kind!=0 || f.a->num_lev==40,"dense fixture lost deferred dependencies");
    require(kind!=1 || (f.fill.size()==11 && f.a->nnz==99),"sparse fixture lost fill");
    require(kind!=2 || f.a->nnz==5,"diagonal fixture changed");
    if(kind==3) {
        const auto plan=make_panel_update_plan(*f.a,32,false,true);
        require(plan.groups.size()>uint64_t(f.a->num_lev)*1024,"wide fixture lost target parallelism");
    }
    return f;
}
std::vector<REAL> input(const Fixture& f,unsigned round,size_t task,bool poison=false) {
    const auto& a=*f.a; auto values=a.val;
    for(unsigned c=0;c<a.n;++c)for(unsigned p=a.sym_c_ptr[c];p<a.sym_c_ptr[c+1];++p) {
        const unsigned r=a.sym_r_idx[p];
        values[p]*=(REAL(1)+REAL((r+round+task)%17)/64)*
                   (REAL(1)+REAL((c+2*round+task)%19)/64);
        // Explicit compact entries change both into and out of zero.
        if(r!=c && (p+round)%3==0)values[p]=0;
    }
    for(unsigned p:f.fill)values[p]=poison?REAL(.125)+REAL(task)/1024:REAL(0);
    return values;
}
// Independent dense L*U reconstruction, including structural zero coordinates.
// This does not consume the native plan or the library's residual implementation.
void residual(const Symbolic_Matrix& a,const REAL* source,const REAL* factors) {
    const size_t n=a.n;
    std::vector<long double> original(n*n,0),lower(n*n,0),upper(n*n,0);
    for(size_t c=0;c<n;++c) {
        lower[c*n+c]=1;
        for(unsigned p=a.sym_c_ptr[c];p<a.sym_c_ptr[c+1];++p) {
            const unsigned r=a.sym_r_idx[p];
            require(std::isfinite(factors[p]),"nonfinite factor");
            original[r*n+c]=source[p];
            (r>c?lower:upper)[r*n+c]=factors[p];
        }
    }
    long double error=0,scale=0;
    for(size_t r=0;r<n;++r) {
        long double row_error=0,row_scale=0;
        for(size_t c=0;c<n;++c) {
            long double product=0;
            for(size_t k=0;k<n;++k)product+=lower[r*n+k]*upper[k*n+c];
            row_error+=std::abs(product-original[r*n+c]);
            row_scale+=std::abs(original[r*n+c]);
        }
        error=std::max(error,row_error);scale=std::max(scale,row_scale);
    }
    require(scale>0 && error/scale<(sizeof(REAL)==8?2e-12L:2e-5L),"independent L*U residual failed");
}
void check(GpuTaskSet& set,const std::vector<Fixture>& fixtures,
           const std::vector<size_t>& identities,const std::vector<REAL>& source) {
    set.check();
    for(size_t t=0;t<identities.size();++t) {
        require(set.factors(t)==set.factors(0)+set.factor_offset(t),"output is not in logical task order");
        residual(*fixtures[identities[t]].a,source.data()+set.factor_offset(t),set.factors(t));
    }
}
void lifecycle(bool wide) {
    std::vector<Fixture> fixtures;
    for(unsigned kind=0;kind<3;++kind)fixtures.push_back(fixture(wide && kind==0 ? 3 : kind));
    std::vector<size_t> identities;
    for(size_t t=0;t<(wide?5:33);++t)identities.push_back((t*7+t/3+2)%3);
    std::unique_ptr<GpuTaskSet> set;
    {
        std::vector<std::unique_ptr<GpuMatrix>> matrices;
        std::vector<const GpuMatrix*> templates;
        for(const auto& f:fixtures) {
            matrices.push_back(std::make_unique<GpuMatrix>(*f.a,f.positions,1,false,UpdateMode::Fused));
            templates.push_back(matrices.back().get());
        }
        rejects<std::invalid_argument>([&]{GpuTaskSet invalid(templates,{});},"empty set accepted");
        rejects<std::out_of_range>([&]{GpuTaskSet invalid(templates,{3});},"invalid template index accepted");
        rejects<std::invalid_argument>([&]{GpuTaskSet invalid({nullptr},{0});},"null template accepted");
        set=std::make_unique<GpuTaskSet>(templates,identities);
        // Upload-only destruction must drain while its source is still alive.
        const auto source=input(fixtures[0],1,0);
        {GpuTaskSet abandoned(templates,{0});abandoned.upload_input(source.data(),source.size());}
    } // Set alone retains the plans and immutable input snapshots.
    require(set->task_count()==identities.size(),"wrong task count");
    size_t full_count=0,compact_count=0;
    std::vector<REAL> defaults;
    for(size_t t=0;t<identities.size();++t) {
        const auto& f=fixtures[identities[t]];
        require(set->factor_offset(t)==full_count && set->compact_input_offset(t)==compact_count,"wrong input prefix");
        full_count+=f.a->nnz;compact_count+=f.positions.size();
        defaults.insert(defaults.end(),f.a->val.begin(),f.a->val.end());
    }
    require(set->factor_offset(identities.size())==full_count &&
            set->compact_input_offset(identities.size())==compact_count,"wrong terminal prefix");
    rejects<std::out_of_range>([&]{set->factor_offset(identities.size()+1);},"invalid offset accepted");
    rejects<std::out_of_range>([&]{set->factors(identities.size());},"invalid result index accepted");
    rejects<std::logic_error>([&]{set->factors(0);},"uninitialized result accepted");
    const size_t initial_pinned=set->pinned_bytes();
    require(initial_pinned==full_count*sizeof(REAL),"default cache allocated eagerly");
    for(unsigned round=0;round<3;++round) {
        std::vector<REAL> full,compact;
        for(size_t t=0;t<identities.size();++t) {
            const auto& f=fixtures[identities[t]];const auto one=input(f,round,t,round==0);
            full.insert(full.end(),one.begin(),one.end());
            for(unsigned p:f.positions)compact.push_back(one[p]);
        }
        if(round==0) {
            rejects<std::invalid_argument>([&]{set->upload_input(full.data(),full.size()-1);},"wrong full size accepted");
            rejects<std::invalid_argument>([&]{set->upload_compact_input(nullptr,compact.size());},"null compact accepted");
            set->upload_input(full.data(),full.size());
        } else set->upload_compact_input(compact.data(),compact.size());
        rejects<std::logic_error>([&]{set->factors(0);},"upload did not invalidate result");
        rejects<std::logic_error>([&]{set->wait();},"upload-only wait accepted");
        rejects<std::logic_error>([&]{set->upload_input();},"second queued input accepted");
        set->enqueue();
        rejects<std::logic_error>([&]{set->enqueue();},"second pending execution accepted");
        rejects<std::logic_error>([&]{set->check();},"pending result accepted");
        set->wait();set->wait();check(*set,fixtures,identities,full);
        require(set->pinned_bytes()==initial_pinned,"fresh path allocated default cache");
    }
    set->upload_input();set->enqueue();set->wait();check(*set,fixtures,identities,defaults);
    require(set->pinned_bytes()==initial_pinned+full_count*sizeof(REAL),"default cache size incorrect");
    const double setup_ms=set->default_input_setup_ms();
    set->enqueue();set->wait();check(*set,fixtures,identities,defaults);
    require(set->default_input_setup_ms()==setup_ms,"default cache rebuilt");
}
void api_boundaries() {
    auto sparse=fixture(1),diagonal=fixture(2);
    {
        GpuMatrix panel(*diagonal.a,1,false,UpdateMode::Panel);
        rejects<std::invalid_argument>([&]{GpuTaskSet invalid({&panel},{0});},"non-Fused template accepted");
    }
    GpuMatrix compact_template(*sparse.a,sparse.positions,1,false,UpdateMode::Fused);
    {
        GpuMatrix full_only_template(*diagonal.a,1,false,UpdateMode::Fused);
        GpuTaskSet set({&full_only_template,&compact_template},{1,0});
        auto full=input(sparse,1,0);const auto second=input(diagonal,1,1);
        full.insert(full.end(),second.begin(),second.end());
        std::vector<REAL> compact(sparse.positions.size(),REAL(1));
        rejects<std::invalid_argument>([&]{set.upload_compact_input(compact.data(),compact.size());},
                                      "partially mapped collection accepted compact input");
        set.upload_input(full.data(),full.size());set.enqueue();set.wait();set.check();
        residual(*sparse.a,full.data(),set.factors(0));
        residual(*diagonal.a,second.data(),set.factors(1));
        set.enqueue();set.wait();set.check();
        residual(*sparse.a,sparse.a->val.data(),set.factors(0));
        residual(*diagonal.a,diagonal.a->val.data(),set.factors(1));
    }
    // Known static perturbation: a diagonal matrix with one nonzero pivot.
    // Mixed descriptors must apply the threshold only to the perturbed template.
    std::fill(diagonal.a->val.begin(),diagonal.a->val.end(),REAL(0));
    diagonal.a->val[2]=REAL(2);
    auto effective=diagonal.a->val;
    for(auto& value:effective)if(value==0)value=REAL(2)*REAL(3.45e-4f);
    GpuMatrix perturbed(*diagonal.a,diagonal.positions,1,true,UpdateMode::Fused);
    GpuTaskSet set({&compact_template,&perturbed},{1,0,1});
    std::vector<REAL> full(set.factor_offset(set.task_count()),REAL(1));
    std::vector<REAL> compact(set.compact_input_offset(set.task_count()),REAL(1));
    rejects<std::invalid_argument>([&]{set.upload_input(full.data(),full.size());},"perturbed set accepted full fresh input");
    rejects<std::invalid_argument>([&]{set.upload_compact_input(compact.data(),compact.size());},"perturbed set accepted compact input");
    for(unsigned round=0;round<2;++round) {
        set.enqueue();set.wait();set.check();
        residual(*diagonal.a,effective.data(),set.factors(0));
        residual(*sparse.a,sparse.a->val.data(),set.factors(1));
        residual(*diagonal.a,effective.data(),set.factors(2));
    }
}
} // namespace

int main() {
    try {
        lifecycle(false);lifecycle(true);api_boundaries();
        std::cout<<"PASS heterogeneous task-set offsets, compact reset, snapshots, lifetime and API boundaries\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"Task-set regression failed: "<<error.what()<<'\n';return 1;
    }
}
