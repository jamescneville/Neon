#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "../Index_3d.h"
#include "Neon/domain/details/mGrid/mPartition.h"
// #include "./bIndex.h"
#include "../ngh_idx.h"

// NOTE: we need this header to avoid errors about missing copy constructor for Pitch (Vec_4d)
// #include "Neon/core/types/vec/vec4d_integer.timp.h"

namespace wp {

// Fixed-point encodings for a DEDICATED integer fine-to-coarse restriction
// field. Population fields remain valid FP32/FP64 values at all times.
//
// FP32 population storage -> int64 accumulator, scale 2^48.
//   * raw-sum range: approximately [-32768, 32768)
//   * FP32 values with exponent >= -25 (|x| >= 2^-25) encode exactly
//   * smaller values have absolute quantization <= 2^-49
//
// FP64 population storage -> int64 accumulator, scale 2^59, range [-16, 16).
//
// The range applies to the raw sum before the existing raw-last 1/N factor.
static constexpr double kNeonMresFixedScaleFloat32 = 281474976710656.0;     // 2^48
static constexpr double kNeonMresFixedScaleFloat64 = 576460752303423488.0;  // 2^59
static_assert(sizeof(int64_t) == 8, "deterministic restriction requires int64");
static_assert(sizeof(int64_t) == sizeof(long long),
              "deterministic FP64 restriction requires a 64-bit CUDA signed word");
static_assert(sizeof(unsigned long long) == 8,
              "deterministic FP64 restriction requires a 64-bit CUDA atomic word");

// NOTE: We create a subclass so that we can add a custom constructor
template <typename T>
class NeonMultiresPartition : public ::Neon::domain::details::mGrid::mPartition<T, 0, Neon::domain::details::StaticBlockDefault>
{
   public:
    // initialize from bytes
    NeonMultiresPartition(const char* bytes, size_t n)
    {
        // printf("NeonMultiresPartition(const char* bytes, size_t n) n %ld sizeof(*this) %ld\n", n, sizeof(*this));
        assert(n == sizeof(*this));
        memcpy(this, bytes, n);
    }

    // NOTE: need default constructor for adjoint vars
    NeonMultiresPartition()
    {
    }
};

using NeonMultiresPartition_int8 = NeonMultiresPartition<int8_t>;
using NeonMultiresPartition_uint8 = NeonMultiresPartition<uint8_t>;

using NeonMultiresPartition_int32 = NeonMultiresPartition<int32_t>;
using NeonMultiresPartition_uint32 = NeonMultiresPartition<uint32_t>;

using NeonMultiresPartition_int64 = NeonMultiresPartition<int64_t>;
using NeonMultiresPartition_uint64 = NeonMultiresPartition<uint64_t>;

using NeonMultiresPartition_float32 = NeonMultiresPartition<float>;
using NeonMultiresPartition_float64 = NeonMultiresPartition<double>;


template <typename T>
CUDA_CALLABLE inline auto neon_read(
    NeonMultiresPartition<T>& p,
    NeonBlockIdx const&       idx,
    int                       card)
    -> T
{
    return p(idx, card);
}

template <typename T>
CUDA_CALLABLE inline auto neon_write(
    NeonMultiresPartition<T>& p,
    NeonBlockIdx const&       idx,
    int                       card,
    T const&                  value)
    -> void
{
    p(idx, card) = value;
}

template <typename T>
CUDA_CALLABLE inline auto neon_cardinality(
    NeonMultiresPartition<T>& p)
    -> int
{
    return p.cardinality();
}

template <typename T>
CUDA_CALLABLE inline auto neon_read_ngh(
    const NeonMultiresPartition<T>& p,
    NeonBlockIdx const&             idx,
    NeonNghIdx const&               ngh,
    int                             card,
    T                               alternative,
    bool&                           valid) -> T
{
    typename NeonMultiresPartition<T>::NghData nghData =
        p.getNghData(idx, ngh, card, alternative);
    valid = nghData.isValid();
    return nghData.getData();
}

template <typename T>
CUDA_CALLABLE inline auto neon_write_ngh(
    NeonMultiresPartition<T>& p,
    NeonBlockIdx const&       idx,
    NeonNghIdx const&         ngh,
    int                       card,
    T                         value) -> bool
{
    const auto nghIdx = p.helpGetNghIdx(idx, ngh);
    auto const isActive = p.isActive(nghIdx);
    if (isActive) {
        p(nghIdx, card) = value;
    }
    return isActive;
}

template <typename T>
CUDA_CALLABLE inline auto neon_lbm_read_coarser_ngh(
    const NeonMultiresPartition<T>& p,
    NeonBlockIdx const&             idx,
    NeonNghIdx const&               ngh,
    int                             card,
    T                               alternative,
    bool&                           valid) -> T
{
    auto getUncleOffset = [](const Neon::int8_3d& cell,
                             const Neon::int8_3d& q) -> Neon::int8_3d {  // uncleOffset
        // given a local index within a cell and a population direction (q)
        // find the uncle's (the parent neighbor) offset from which he desired population (q) should be read
        // this offset is wrt the cell containing the localID (i.e., the parent of localID)
        auto off = [](const int8_t i, const int8_t j) -> int8_t {
            // 0, -1 --> -1
            // 1, -1 --> 0
            // 0, 0 --> 0
            // 0, 1 --> 0
            // 1, 1 --> 1
            const int8_t s = i + j;
            return (s <= 0) ? s : s - 1;
        };

        Neon::int8_3d offset(off(cell.x % Neon::domain::details::StaticBlockDefault::userBlockSizeX, q.x),
                             off(cell.y % Neon::domain::details::StaticBlockDefault::userBlockSizeY, q.y),
                             off(cell.z % Neon::domain::details::StaticBlockDefault::userBlockSizeZ, q.z));
        return offset;
    };
    Neon::int8_3d uncleDir = getUncleOffset(idx.mInDataBlockIdx, ngh);
    const auto    uncleData = p.uncleVal(idx, uncleDir, card, alternative);
    valid = uncleData.isValid();
    return uncleData.getData();
}


template <typename T>
CUDA_CALLABLE inline auto neon_mres_lbm_store_op(const NeonMultiresPartition<T>& pout,
                                                 const NeonBlockIdx&             cell,
                                                 const int8_t                    q,
                                                 NeonNghIdx const&               qDir,
                                                 const T                         cellVal) -> void
{

    auto uncleOffset = [](const auto& cell, const Neon::int8_3d& q) -> Neon::int8_3d {
        // given a local index within a cell and a population direction (q)
        // find the uncle's (the parent neighbor) offset from which the desired population (q) should be read
        // this offset is wrt the cell containing the localID (i.e., the parent of localID)
        auto off = [](const int8_t i, const int8_t j) {
            // 0, -1 --> -1
            // 1, -1 --> 0
            // 0, 0 --> 0
            // 0, 1 --> 0
            // 1, 1 --> 1
            const int8_t s = i + j;
            return (s <= 0) ? s : s - 1;
        };
        Neon::int8_3d offset(off(cell.x % Neon::domain::details::StaticBlockDefault::userBlockSizeX, q.x),
                             off(cell.y % Neon::domain::details::StaticBlockDefault::userBlockSizeY, q.y),
                             off(cell.z % Neon::domain::details::StaticBlockDefault::userBlockSizeZ, q.z));
        return offset;
    };

    if (qDir.x == 0 && qDir.y == 0 && qDir.z == 0) {
        return;
    }

    const Neon::int8_3d uncleDir = uncleOffset(cell.mInDataBlockIdx, qDir);

    // we try to access a cell on the same level (i.e., the refined level) along the same
    // direction as the uncle and we use this a proxy to check if there is an unrefined uncle
    const auto cn = pout.helpGetNghIdx(cell, uncleDir);

    // cn may not be active because 1. it is outside the domain, or 2. this location is occupied by a coarse cell
    // we are interested in 2.
    if (!pout.isActive(cn)) {

        // now, we can get the uncle but we need to make sure it is active i.e.,
        // it is not out side the domain boundary
        const auto uncle = pout.getUncle(cell, uncleDir);
        if (uncle.isActive()) {

            // locate the coarse cell where we should store this cell info
            const Neon::int8_3d CsDir = uncleDir - qDir;

            const auto cs = pout.getUncle(cell, CsDir);

            const auto csChild = pout.helpGetNghIdx(cell, CsDir);

            if (cs.isActive() && pout.isActive(csChild)) {
                atomicAdd(&pout.uncleVal(cell, CsDir, q), cellVal);
            }
        }
    }
}

// ============================================================================
// Dedicated integer deterministic restriction
// ============================================================================
//
// The accumulator is an int32/int64 mField with the same multires topology as
// the population fields.  Fine cells atomically deposit quantized post-collision
// populations into the accumulator's coarse uncle halo.  The coarse stream reads
// and decodes that integer field; f_0/f_1 never contain encoded integer bits.
//
// Integer atomic addition is order-independent.  Checked CAS loops prevent
// silent signed overflow.  A range violation traps on CUDA rather than wrapping.

CUDA_CALLABLE inline void neon_mres_fixed_range_error()
{
#ifdef __CUDA_ARCH__
    asm volatile("trap;");
#else
    assert(false && "deterministic restriction fixed-point range exceeded");
#endif
}

CUDA_CALLABLE inline auto neon_mres_uncle_offset(
    const NeonBlockIdx& cell,
    const NeonNghIdx&   qDir) -> Neon::int8_3d
{
    auto off = [](const int8_t i, const int8_t j) {
        const int8_t s = i + j;
        return (s <= 0) ? s : s - 1;
    };
    return Neon::int8_3d(
        off(cell.mInDataBlockIdx.x % Neon::domain::details::StaticBlockDefault::userBlockSizeX, qDir.x),
        off(cell.mInDataBlockIdx.y % Neon::domain::details::StaticBlockDefault::userBlockSizeY, qDir.y),
        off(cell.mInDataBlockIdx.z % Neon::domain::details::StaticBlockDefault::userBlockSizeZ, qDir.z));
}

template <typename T>
CUDA_CALLABLE inline auto neon_mres_has_restriction_target(
    const NeonMultiresPartition<T>& p,
    const NeonBlockIdx&             cell,
    const NeonNghIdx&               qDir,
    Neon::int8_3d&                  csDir) -> bool
{
    if (qDir.x == 0 && qDir.y == 0 && qDir.z == 0) {
        return false;
    }

    const Neon::int8_3d uncleDir = neon_mres_uncle_offset(cell, qDir);
    const auto          cn = p.helpGetNghIdx(cell, uncleDir);
    if (p.isActive(cn)) {
        return false;
    }

    const auto uncle = p.getUncle(cell, uncleDir);
    if (!uncle.isActive()) {
        return false;
    }

    csDir = uncleDir - qDir;
    const auto cs = p.getUncle(cell, csDir);
    const auto csChild = p.helpGetNghIdx(cell, csDir);
    return cs.isActive() && p.isActive(csChild);
}

CUDA_CALLABLE inline void neon_mres_atomic_add_checked(int32_t* address, int32_t deposit)
{
#ifdef __CUDA_ARCH__
    unsigned int* const words = reinterpret_cast<unsigned int*>(address);
    unsigned int observed = atomicCAS(words, 0u, 0u);
    while (true) {
        union Word32 {
            unsigned int u;
            int32_t      s;
        } old_word, new_word;
        old_word.u = observed;
        const long long sum = static_cast<long long>(old_word.s) +
                              static_cast<long long>(deposit);
        if (sum < -2147483648LL || sum > 2147483647LL) {
            neon_mres_fixed_range_error();
            return;
        }
        new_word.s = static_cast<int32_t>(sum);
        const unsigned int prior = atomicCAS(words, observed, new_word.u);
        if (prior == observed) {
            return;
        }
        observed = prior;
    }
#else
    const long long sum = static_cast<long long>(*address) +
                          static_cast<long long>(deposit);
    if (sum < -2147483648LL || sum > 2147483647LL) {
        neon_mres_fixed_range_error();
        return;
    }
    *address = static_cast<int32_t>(sum);
#endif
}

CUDA_CALLABLE inline void neon_mres_atomic_add_checked(int64_t* address, int64_t deposit)
{
#ifdef __CUDA_ARCH__
    unsigned long long* const words =
        reinterpret_cast<unsigned long long*>(address);
    unsigned long long observed = atomicCAS(words, 0ULL, 0ULL);
    while (true) {
        union Word64 {
            unsigned long long u;
            long long          s;
        } old_word, new_word;
        old_word.u = observed;
        const long long old_value = old_word.s;
        const long long dep = static_cast<long long>(deposit);
        if ((dep > 0 && old_value > 9223372036854775807LL - dep) ||
            (dep < 0 && old_value < (-9223372036854775807LL - 1LL) - dep)) {
            neon_mres_fixed_range_error();
            return;
        }
        new_word.s = old_value + dep;
        const unsigned long long prior = atomicCAS(words, observed, new_word.u);
        if (prior == observed) {
            return;
        }
        observed = prior;
    }
#else
    const long long old_value = static_cast<long long>(*address);
    const long long dep = static_cast<long long>(deposit);
    if ((dep > 0 && old_value > 9223372036854775807LL - dep) ||
        (dep < 0 && old_value < (-9223372036854775807LL - 1LL) - dep)) {
        neon_mres_fixed_range_error();
        return;
    }
    *address = static_cast<int64_t>(old_value + dep);
#endif
}

CUDA_CALLABLE inline auto neon_mres_lbm_zero_fixed(
    const NeonMultiresPartition<int64_t>& p,
    const NeonBlockIdx&                   cell,
    const int8_t                          q,
    const NeonNghIdx&                     qDir) -> void
{
    Neon::int8_3d csDir(0, 0, 0);
    if (!neon_mres_has_restriction_target(p, cell, qDir, csDir)) {
        return;
    }
    int64_t* const slot = &p.uncleVal(cell, csDir, q);
#ifdef __CUDA_ARCH__
    atomicExch(reinterpret_cast<unsigned long long*>(slot), 0ULL);
#else
    *slot = 0;
#endif
}

CUDA_CALLABLE inline auto neon_mres_lbm_store_fixed_f32(
    const NeonMultiresPartition<int64_t>& p,
    const NeonBlockIdx&                   cell,
    const int8_t                          q,
    const NeonNghIdx&                     qDir,
    const float                           cellVal) -> void
{
    Neon::int8_3d csDir(0, 0, 0);
    if (!neon_mres_has_restriction_target(p, cell, qDir, csDir)) {
        return;
    }

    // Convert the stored FP32 value to FP64 before applying the power-of-two
    // scale.  For normal LBM populations this is an exact integer mapping, not
    // a lossy per-deposit quantization as in the former int32/2^27 path.
    if (!(cellVal > -4.0f && cellVal < 4.0f)) {
        neon_mres_fixed_range_error();
        return;
    }
    const double scaled = static_cast<double>(cellVal) *
                          kNeonMresFixedScaleFloat32;
#ifdef __CUDA_ARCH__
    const int64_t fixed = static_cast<int64_t>(__double2ll_rn(scaled));
#else
    const int64_t fixed = static_cast<int64_t>(llrint(scaled));
#endif
    neon_mres_atomic_add_checked(&p.uncleVal(cell, csDir, q), fixed);
}

CUDA_CALLABLE inline auto neon_mres_lbm_store_fixed_f64(
    const NeonMultiresPartition<int64_t>& p,
    const NeonBlockIdx&                   cell,
    const int8_t                          q,
    const NeonNghIdx&                     qDir,
    const double                          cellVal) -> void
{
    Neon::int8_3d csDir(0, 0, 0);
    if (!neon_mres_has_restriction_target(p, cell, qDir, csDir)) {
        return;
    }
    if (!(cellVal > -16.0 && cellVal < 16.0)) {
        neon_mres_fixed_range_error();
        return;
    }
    const double scaled = cellVal * kNeonMresFixedScaleFloat64;
#ifdef __CUDA_ARCH__
    const int64_t fixed = static_cast<int64_t>(__double2ll_rn(scaled));
#else
    const int64_t fixed = static_cast<int64_t>(llrint(scaled));
#endif
    neon_mres_atomic_add_checked(&p.uncleVal(cell, csDir, q), fixed);
}

CUDA_CALLABLE inline auto neon_mres_lbm_decode_fixed_f32(
    const int64_t fixed) -> double
{
    return static_cast<double>(fixed) / kNeonMresFixedScaleFloat32;
}

CUDA_CALLABLE inline auto neon_mres_lbm_decode_fixed_f64(
    const int64_t fixed) -> double
{
    return static_cast<double>(fixed) / kNeonMresFixedScaleFloat64;
}

template <typename T>
CUDA_CALLABLE inline auto neon_is_active(
    const NeonMultiresPartition<T>& p,
    NeonBlockIdx const&             idx,
    NeonNghIdx const&               ngh) -> bool
{
    bool status = p.isValid(idx, ngh);
    return status;
}

template <typename T>
CUDA_CALLABLE inline auto neon_ngh_idx(
    const NeonMultiresPartition<T>& p,
    NeonBlockIdx const&             idx,
    NeonNghIdx const&               ngh) -> NeonBlockIdx
{
    NeonBlockIdx const ngh_idx = p.helpGetNghIdx(idx, ngh);
    return ngh_idx;
}

template <typename T>
CUDA_CALLABLE inline auto neon_partition_id(
    NeonMultiresPartition<T>& p)
    -> int
{
    return p.prtID();
}

// print
// template<typename T>
// CUDA_CALLABLE inline auto neon_print_dbg(const NeonMultiresPartition<T>& p) -> void
//{
//   const Neon::index_3d& dim = p.dim();
//   const Neon::index_3d& halo = p.halo();
//   const Neon::index_3d& origin = p.origin();
//   const int prtID = p.prtID();
//   printf("NeonMultiresPartition(dim={%d, %d, %d}, halo={%d, %d, %d}, origin={%d, %d, %d}, mem=%p prtID %d)\n",
//      dim.x, dim.y, dim.z,
//      halo.x, halo.y, halo.z,
//      origin.x, origin.y, origin.z,
//      p.mem(),
//      prtID
//   );
//}

template <typename T>
CUDA_CALLABLE inline auto neon_global_idx(
    NeonMultiresPartition<T>& p,
    NeonBlockIdx const&       idx)
    -> Neon::index_3d
{
    Neon::index_3d globalIdx = p.getGlobalIndex(idx);
    return globalIdx;
}
//// Multi-res Capabilities

template <typename T>
CUDA_CALLABLE inline auto neon_read_child(
    NeonMultiresPartition<T>& p,
    const NeonBlockIdx&       parentCell,
    const NeonNghIdx          child,
    int                       card,
    const T&                  alternativeVal,
    bool&                     isValid) -> T
{

    typename NeonMultiresPartition<T>::NghData nghData =
        p.childVal(parentCell, child, card, alternativeVal);
    isValid = nghData.isValid();
    return nghData.getData();
}

template <typename T>
CUDA_CALLABLE inline auto neon_get_child(
    NeonMultiresPartition<T>& p,
    const NeonBlockIdx&       parentCell,
    const NeonNghIdx          child) -> NeonNghIdx
{
    return p.getChild(parentCell, child);
}

template <typename T>
CUDA_CALLABLE inline auto neon_read_child(
    NeonMultiresPartition<T>& p,
    const NeonBlockIdx&       childIdx,
    int                       card) -> T
{
    return p.childVal(childIdx, card);
}

template <typename T>
CUDA_CALLABLE inline auto neon_has_child(
    NeonMultiresPartition<T>& p,
    const NeonBlockIdx&       idx) -> bool
{
    return p.hasChildren(idx);
}

template <typename T>
CUDA_CALLABLE inline auto neon_has_finer_ngh(
    NeonMultiresPartition<T>& p,
    const NeonBlockIdx&       cell,
    const NeonNghIdx          nghDir) -> bool
{
    return p.hasChildren(cell, nghDir);
}

template <typename T>
CUDA_CALLABLE inline auto neon_getParent(
    NeonMultiresPartition<T>& p,
    const NeonBlockIdx&       cell) -> NeonBlockIdx
{
    return p.getParent(cell);
}

template <typename T>
CUDA_CALLABLE inline auto neon_read_parent(
    NeonMultiresPartition<T> const& p,
    const NeonBlockIdx&             cell,
    int                             card) -> T
{
    return p.parentVal(cell, card);
}


template <typename T>
CUDA_CALLABLE inline auto neon_write_parent(
    NeonMultiresPartition<T> const& p,
    const NeonBlockIdx&             cell,
    int                             card,
    T                               value) -> void
{
    return p.parentVal(cell, card) = value;
}

// template <typename T>
// CUDA_CALLABLE inline auto neon_parentVal_atomic_write(
//     NeonMultiresPartition<T> const& p,
//     const NeonBlockIdx&             cell,
//     int                             card,
//     T value) -> void
//{
//     atomicAdd(&p.parentVal(cell, card), value);
// }

template <typename T>
CUDA_CALLABLE inline auto neon_has_parent(
    NeonMultiresPartition<T> const& p,
    const NeonBlockIdx&             cell) -> bool
{
    return p.hasParent(cell);
}

template <typename T>
CUDA_CALLABLE inline auto neon_getUncle(
    NeonMultiresPartition<T> const& p,
    const NeonBlockIdx&             cell,
    const NeonNghIdx                direction) -> NeonBlockIdx
{
    return p.getUncle(cell, direction);
}

template <typename T>
CUDA_CALLABLE inline auto neon_read_coarser_ngh(
    NeonMultiresPartition<T> const& p,
    const NeonBlockIdx&             cell,
    const NeonNghIdx                direction,
    int                             card,
    T                               alternativeValue,
    bool&                           isValid) -> T
{
    typename NeonMultiresPartition<T>::NghData nghData =
        p.uncleVal(cell, direction, card, alternativeValue);
    isValid = nghData.isValid();
    return nghData.getData();
}

template <typename T>
CUDA_CALLABLE inline auto neon_read_coarser_ngh(
    NeonMultiresPartition<T> const& p,
    const NeonBlockIdx&             cell,
    const NeonNghIdx                direction,
    int                             card) -> T
{
    return p.uncleVal(cell, direction, card);
}

template <typename T>
CUDA_CALLABLE inline auto neon_refinement_factor(NeonMultiresPartition<T> const& p,
                                                 int                             level) -> int
{
    return p.getRefFactor(level);
}

template <typename T>
CUDA_CALLABLE inline auto neon_spacing(NeonMultiresPartition<T> const& p,
                                       int                             level) -> int
{
    return p.getSpacing(level);
}

template <typename T>
CUDA_CALLABLE inline auto neon_level(NeonMultiresPartition<T> const& p) -> int
{
    return p.getLevel();
}

template <typename T>
CUDA_CALLABLE inline auto neon_print_log(NeonMultiresPartition<T> const& p, bool masterOnly) -> void
{
    return p.printLog(masterOnly);
}

}  // namespace wp
