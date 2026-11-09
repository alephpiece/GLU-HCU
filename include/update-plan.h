#ifndef GLU_UPDATE_PLAN_H
#define GLU_UPDATE_PLAN_H
#include "symbolic.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

// Structural panel plan. Every entry is a multiply-subtract; stable destination
// sorting changes ownership, never the contribution set.
struct UpdateContribution { unsigned destination, lower, pivot; };
struct NormalizedUpdatePair { unsigned lower, pivot; };
// With inline singletons enabled, the destination high bit marks an inline
// lower/pivot pair in begin/end. Templates outside this range keep normal pairs.
constexpr unsigned inline_update_pair_bit=1u<<31;
struct UpdateGroup { unsigned destination, begin, end; };
static_assert(sizeof(UpdateGroup)==12 && sizeof(NormalizedUpdatePair)==8,"update plan record layout changed");
inline std::vector<UpdateContribution> level_contributions(const Symbolic_Matrix& a,unsigned level) {
    std::vector<UpdateContribution> out;
    for(int k=a.level_ptr[level];k<a.level_ptr[level+1];++k) {
        const unsigned source=unsigned(a.level_idx[k]);
        for(unsigned q=a.csr_diag_ptr[source]+1;q<a.csr_r_ptr[source+1];++q) {
            const unsigned target=a.csr_c_idx[q];
            unsigned at=a.sym_c_ptr[target],end=a.sym_c_ptr[target+1];
            while(at<end && a.sym_r_idx[at]<source) ++at;
            if(at==end || a.sym_r_idx[at]!=source) throw std::runtime_error("aggregation missing pivot");
            const unsigned pivot=at++;
            for(unsigned l=a.l_col_ptr[source]+1;l<a.sym_c_ptr[source+1];++l) {
                while(at<end && a.sym_r_idx[at]<a.sym_r_idx[l]) ++at;
                if(at==end || a.sym_r_idx[at]!=a.sym_r_idx[l]) throw std::runtime_error("aggregation missing lower row");
                out.push_back({at,l,pivot});
            }
        }
    }
    return out;
}
inline void append_update_groups(std::vector<UpdateContribution>& ops,std::vector<UpdateGroup>& groups,
                                 std::vector<NormalizedUpdatePair>& pairs,bool inline_singletons=false) {
    std::stable_sort(ops.begin(),ops.end(),[](const auto& x,const auto& y){return x.destination<y.destination;});
    if(ops.size()>std::numeric_limits<unsigned>::max()-pairs.size()) throw std::overflow_error("aggregation pair capacity");
    for(size_t first=0;first<ops.size();) {
        size_t last=first+1;
        while(last<ops.size() && ops[last].destination==ops[first].destination) ++last;
        if(groups.size()==std::numeric_limits<unsigned>::max()) throw std::overflow_error("aggregation group capacity");
        if(inline_singletons && last-first==1) {
            const auto u=ops[first];
            if(u.destination>=inline_update_pair_bit)throw std::logic_error("inline update destination range");
            groups.push_back({u.destination|inline_update_pair_bit,u.lower,u.pivot});
        } else {
            const unsigned begin=unsigned(pairs.size());
            for(size_t i=first;i<last;++i)pairs.push_back({ops[i].lower,ops[i].pivot});
            groups.push_back({ops[first].destination,begin,unsigned(pairs.size())});
        }
        first=last;
    }
}
struct UpdateRange { unsigned begin,end; };
struct DenseUpdateTile {
    unsigned dst_begin, l_begin, u_begin, nrows, ncols, length, row_begin, col_begin;
};
static_assert(sizeof(unsigned) == 4, "dense update indices require 32-bit unsigned");
static_assert(sizeof(DenseUpdateTile) == 32, "dense tile layout changed");

// The input contains the complete deferred contributions of ONE dependency
// window. Partition whole destinations; never split a destination between the
// dense and remaining paths. Stable sorting preserves its contribution order.
// D is column-major [column*nrows+row], L [q*nrows+row], U [q*ncols+column].
// Only D may contain UINT_MAX holes. length is the plain reduction length,
// with NO masked-tile flag; consumers must test D against UINT_MAX.
// All 16x16 views of a rectangle share its single index allocation.
// On a capacity/allocation exception the caller must discard the partial plan.
inline void extract_dense_updates(const Symbolic_Matrix& a,
        std::vector<UpdateContribution>& deferred,
        std::vector<DenseUpdateTile>& tiles, std::vector<unsigned>& indices) {
    constexpr unsigned missing = std::numeric_limits<unsigned>::max();
    constexpr size_t absent = std::numeric_limits<size_t>::max();
    const uint64_t capacity = missing;
    if (indices.size() > capacity || tiles.size() > capacity)
        throw std::overflow_error("dense update plan capacity");
    if (deferred.empty()) return;
    const size_t original_products = deferred.size();
    if (a.sym_c_ptr.size() != size_t(a.n) + 1 || a.sym_r_idx.size() != a.nnz
        || a.sym_c_ptr.front() != 0 || a.sym_c_ptr.back() != a.nnz)
        throw std::runtime_error("dense update symbolic dimensions");
    std::vector<unsigned> column_of(a.nnz);
    for (unsigned c = 0; c < a.n; ++c) {
        const unsigned first = a.sym_c_ptr[c], last = a.sym_c_ptr[c + 1];
        if (first > last || last > a.nnz)
            throw std::runtime_error("dense update invalid column range");
        for (unsigned p = first; p < last; ++p) {
            if (a.sym_r_idx[p] >= a.n || (p > first && a.sym_r_idx[p - 1] >= a.sym_r_idx[p]))
                throw std::runtime_error("dense update requires unique sorted rows");
            column_of[p] = c;
        }
    }
    for (const auto& u : deferred)
        if (u.destination >= a.nnz || u.lower >= a.nnz || u.pivot >= a.nnz)
            throw std::runtime_error("dense update contribution index");
    std::stable_sort(deferred.begin(), deferred.end(), [](const auto& x, const auto& y) {
        return x.destination < y.destination;
    });
    struct Group { size_t begin, end; unsigned destination; };
    std::vector<Group> groups;
    std::map<std::vector<unsigned>, std::vector<size_t>> rectangles;
    for (size_t first = 0; first < deferred.size();) {
        size_t last = first + 1;
        while (last < deferred.size() && deferred[last].destination == deferred[first].destination) ++last;
        const size_t group = groups.size();
        const unsigned dst = deferred[first].destination;
        groups.push_back({first, last, dst});
        if (last - first <= 32) {
            // Equal complete ORDERED pivot lists, within a fixed 128x128 region.
            std::vector<unsigned> key{a.sym_r_idx[dst] / 128, column_of[dst] / 128};
            for (size_t q = first; q < last; ++q) key.push_back(a.sym_r_idx[deferred[q].pivot]);
            rectangles[key].push_back(group);
        }
        first = last;
    }
    std::vector<unsigned char> selected(groups.size(), 0);
    size_t selected_products = 0;
    for (const auto& entry : rectangles) {
        const auto& members = entry.second;
        if (members.size() < 64) continue;
        std::vector<unsigned> rows, cols;
        for (size_t i : members) {
            rows.push_back(a.sym_r_idx[groups[i].destination]);
            cols.push_back(column_of[groups[i].destination]);
        }
        std::sort(rows.begin(), rows.end()); rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        std::sort(cols.begin(), cols.end()); cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        const unsigned nr = unsigned(rows.size()), nc = unsigned(cols.size());
        const unsigned length = unsigned(groups[members.front()].end - groups[members.front()].begin);
        if (nr < 8 || nc < 8 || members.size() * 2 < size_t(nr) * nc) continue;
        // The region key bounds nr/nc by 128 and contribution length by 32.
        std::vector<size_t> grid(size_t(nr) * nc, absent);
        std::vector<unsigned> lower(size_t(nr) * length, missing), pivot(size_t(nc) * length, missing);
        bool valid = true;
        for (size_t i : members) {
            const auto& g = groups[i];
            const unsigned r = unsigned(std::lower_bound(rows.begin(), rows.end(), a.sym_r_idx[g.destination]) - rows.begin());
            const unsigned c = unsigned(std::lower_bound(cols.begin(), cols.end(), column_of[g.destination]) - cols.begin());
            if (grid[size_t(c) * nr + r] != absent) valid = false;
            grid[size_t(c) * nr + r] = i;
            if (g.end - g.begin != length)
                throw std::logic_error("dense update pivot-list length mismatch");
            for (unsigned q = 0; q < length; ++q) {
                const auto& u = deferred[g.begin + q];
                auto& l = lower[size_t(q) * nr + r];
                auto& up = pivot[size_t(q) * nc + c];
                if ((l != missing && l != u.lower) || (up != missing && up != u.pivot)) valid = false;
                l = u.lower; up = u.pivot;
            }
        }
        // Holes are permitted only in D. Every real contribution must agree
        // with a complete, separable row/pivot L and column/pivot U product.
        if (std::find(lower.begin(), lower.end(), missing) != lower.end()
            || std::find(pivot.begin(), pivot.end(), missing) != pivot.end()) valid = false;
        if (!valid) continue;
        const uint64_t extra_indices = uint64_t(nr) * nc + lower.size() + pivot.size();
        const uint64_t extra_tiles = uint64_t((nr + 15) / 16) * ((nc + 15) / 16);
        if (extra_indices > capacity - indices.size() || extra_tiles > capacity - tiles.size())
            throw std::overflow_error("dense update plan capacity");
        DenseUpdateTile tile{unsigned(indices.size()), 0, 0, nr, nc, length, 0, 0};
        for (size_t i : grid) indices.push_back(i == absent ? missing : groups[i].destination);
        tile.l_begin = unsigned(indices.size());
        indices.insert(indices.end(), lower.begin(), lower.end());
        tile.u_begin = unsigned(indices.size());
        indices.insert(indices.end(), pivot.begin(), pivot.end());
        for (unsigned c = 0; c < nc; c += 16)
            for (unsigned r = 0; r < nr; r += 16) {
                auto view = tile; view.row_begin = r; view.col_begin = c;
                tiles.push_back(view);
            }
        for (size_t i : members) {
            if (selected[i]) throw std::logic_error("dense update duplicate destination");
            selected[i] = 1;
            const size_t count = groups[i].end - groups[i].begin;
            if (count > original_products - selected_products)
                throw std::logic_error("dense update selected contribution overflow");
            selected_products += count;
        }
    }
    // Compact in place after all descriptors have consumed the original list.
    // Assignment is forward and preserves each retained destination's order.
    size_t retained = 0;
    for (size_t i = 0; i < groups.size(); ++i)
        if (!selected[i])
            for (size_t p = groups[i].begin; p < groups[i].end; ++p) deferred[retained++] = deferred[p];
    if (retained != original_products - selected_products)
        throw std::logic_error("dense update contribution accounting mismatch");
    deferred.resize(retained);
}

struct PanelUpdatePlan {
    std::vector<UpdateGroup> groups;
    std::vector<NormalizedUpdatePair> pairs;
    std::vector<UpdateRange> immediate,deferred;
    std::vector<DenseUpdateTile> dense_tiles;
    std::vector<unsigned> dense_indices;
    std::vector<UpdateRange> dense_ranges;
    uint64_t dense_products=0;
    uint64_t immediate_products=0,deferred_products=0;
    uint64_t inline_singletons=0;
};
// A grouped-update CTA has at least eight target slots (256 threads / at most
// 32 copy lanes). Align chunks to each range start to retain CTA target sets
// across copy geometries. Preserve singleton slots and equal-length order.
inline void sort_deferred_groups_locally(PanelUpdatePlan& plan,bool inline_singletons) {
    constexpr unsigned width=8;
    for(const auto range:plan.deferred)for(size_t base=range.begin;base<range.end;base+=width) {
        UpdateGroup sorted[width];
        unsigned slots[width],count=0;
        for(unsigned slot=0;slot<width && base+slot<range.end;++slot) {
            const auto group=plan.groups[base+slot];
            if((inline_singletons && (group.destination&inline_update_pair_bit)) ||
               group.end-group.begin<=1)continue;
            slots[count]=slot;
            unsigned at=count++;
            const unsigned length=group.end-group.begin;
            while(at && sorted[at-1].end-sorted[at-1].begin>length) {
                sorted[at]=sorted[at-1];
                --at;
            }
            sorted[at]=group;
        }
        for(unsigned at=0;at<count;++at)plan.groups[base+slots[at]]=sorted[at];
    }
}
// A tagged 64-bit record stores either three 21-bit singleton positions or a
// 21-bit destination, 27-bit pair offset and 15-bit count. Wider templates keep
// their original groups; pair operands and contribution order stay unchanged.
constexpr uint64_t packed_update_single_bit=uint64_t(1)<<63;
constexpr unsigned packed_update_position_mask=(1u<<21)-1;
constexpr unsigned packed_update_pair_mask=(1u<<27)-1;
constexpr unsigned packed_update_count_mask=(1u<<15)-1;
inline uint64_t pack_update_range(unsigned destination,unsigned begin,unsigned count) {
    if(destination>packed_update_position_mask || begin>packed_update_pair_mask ||
       !count || count>packed_update_count_mask ||
       uint64_t(begin)+count>uint64_t(packed_update_pair_mask)+1)
        throw std::invalid_argument("packed update range");
    return uint64_t(destination)|(uint64_t(begin)<<21)|(uint64_t(count)<<48);
}
inline std::vector<uint64_t> pack_update_groups(const PanelUpdatePlan& plan,unsigned nnz,bool inline_singletons) {
    if(nnz>=(1u<<21) || plan.pairs.size()>uint64_t(packed_update_pair_mask)+1)return {};
    // Decide once for the whole template before allocating or encoding records.
    for(const auto& group:plan.groups) {
        const bool single=inline_singletons && (group.destination&inline_update_pair_bit);
        const unsigned destination=single ? group.destination&~inline_update_pair_bit : group.destination;
        if(destination>=nnz)throw std::logic_error("packed update destination range");
        if(!single) {
            if(group.begin>=group.end || group.end>plan.pairs.size())throw std::logic_error("packed update pair range");
            if(group.end-group.begin>packed_update_count_mask)return {};
        }
    }
    std::vector<uint64_t> records;
    records.reserve(plan.groups.size());
    for(const auto& group:plan.groups) {
        const bool single=inline_singletons && (group.destination&inline_update_pair_bit);
        const unsigned destination=single ? group.destination&~inline_update_pair_bit : group.destination;
        if(single || group.end-group.begin==1) {
            const auto pair=single ? NormalizedUpdatePair{group.begin,group.end} : plan.pairs[group.begin];
            if(pair.lower>=nnz || pair.pivot>=nnz)throw std::logic_error("packed update operand range");
            records.push_back(packed_update_single_bit|uint64_t(destination)|(uint64_t(pair.lower)<<21)|(uint64_t(pair.pivot)<<42));
        } else records.push_back(pack_update_range(destination,group.begin,group.end-group.begin));
    }
    return records;
}
// Keep destinations read anywhere in this dependency window current at each
// level. Only unread destinations may be grouped across levels and applied at
// the window end; dense extraction partitions those same deferred contributions.
inline PanelUpdatePlan make_panel_update_plan(const Symbolic_Matrix& a,unsigned window,bool dense=false,bool inline_singletons=false) {
    if(!window)throw std::invalid_argument("zero aggregation window");
    inline_singletons=inline_singletons && a.nnz<inline_update_pair_bit;
    PanelUpdatePlan plan;plan.immediate.resize(a.num_lev);plan.deferred.resize(a.num_lev);plan.dense_ranges.resize(a.num_lev);
    for(unsigned begin=0;begin<unsigned(a.num_lev);begin+=window) {
        const unsigned end=std::min(unsigned(a.num_lev),begin+window);
        std::vector<unsigned char> needed(a.nnz,0);
        std::vector<std::vector<UpdateContribution>> levels;
        for(unsigned lev=begin;lev<end;++lev) {
            levels.push_back(level_contributions(a,lev));
            for(int p=a.level_ptr[lev];p<a.level_ptr[lev+1];++p) {
                const unsigned c=unsigned(a.level_idx[p]);
                for(unsigned q=a.l_col_ptr[c];q<a.sym_c_ptr[c+1];++q)needed[q]=1;
            }
            // The source loop above covers every diagonal and lower operand.
            for(const auto& u:levels.back())needed[u.pivot]=1;
        }
        std::vector<UpdateContribution> delayed;
        for(unsigned lev=begin;lev<end;++lev) {
            std::vector<UpdateContribution> now;
            for(const auto& u:levels[lev-begin]) {
                if(needed[u.destination])now.push_back(u);else delayed.push_back(u);
            }
            plan.immediate_products+=now.size();
            const unsigned first=unsigned(plan.groups.size());append_update_groups(now,plan.groups,plan.pairs,inline_singletons);
            plan.immediate[lev]={first,unsigned(plan.groups.size())};
        }
        plan.deferred_products+=delayed.size();
        const unsigned dense_first=unsigned(plan.dense_tiles.size());
        if(dense) {
            const size_t before=delayed.size();extract_dense_updates(a,delayed,plan.dense_tiles,plan.dense_indices);
            plan.dense_products+=before-delayed.size();
        }
        plan.dense_ranges[end-1]={dense_first,unsigned(plan.dense_tiles.size())};
        const unsigned first=unsigned(plan.groups.size());append_update_groups(delayed,plan.groups,plan.pairs,inline_singletons);
        plan.deferred[end-1]={first,unsigned(plan.groups.size())};
        // Every generated record takes exactly one branch above. Sorting only
        // permutes records; it cannot lose or duplicate an individual product.
    }
    if(inline_singletons)for(const auto& group:plan.groups)plan.inline_singletons+=(group.destination&inline_update_pair_bit)!=0;
    if(plan.pairs.size()+plan.inline_singletons+plan.dense_products!=plan.immediate_products+plan.deferred_products)throw std::runtime_error("aggregation contribution accounting");
    return plan;
}
#endif
