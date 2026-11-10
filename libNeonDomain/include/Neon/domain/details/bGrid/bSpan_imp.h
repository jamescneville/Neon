#if !defined(NEON_WARP_COMPILATION)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif
#include "Neon/domain/details/bGrid/bSpan.h"

namespace Neon::domain::details::bGrid {

template <typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto
bSpan<SBlock>::setAndValidateGPUDevice([[maybe_unused]] Idx& bidx) const -> bool
{
#ifdef NEON_PLACE_CUDA_DEVICE
    if (mBlockList != nullptr) {
        // Class-restricted launch. list[0] = count | (voxel mode ? 0x80000000 : 0).
        const uint32_t header = mBlockList[0];
        const uint32_t count = header & 0x7fffffffu;
        if (header & 0x80000000u) {
            // Voxel list: entry = dataBlock * 64 + (x + 4 * y + 16 * z). Thread (b, t) takes entry b * 64 + t, so
            // active voxels are packed densely into warps; surplus threads exit immediately. The encoding is
            // specific to 4x4x4 data blocks; other block sizes trap instead of decoding garbage.
            if constexpr (SBlock::memBlockSizeX == 4 && SBlock::memBlockSizeY == 4 && SBlock::memBlockSizeZ == 4) {
                const uint32_t gid = blockIdx.x * 64u + (threadIdx.x + 4u * (threadIdx.y + 4u * threadIdx.z));
                if (gid >= count) {
                    bidx.mDataBlockIdx = getInvalidBlockId();
                    return false;
                }
                const uint32_t entry = mBlockList[1 + gid];
                bidx.mDataBlockIdx = entry >> 6;
                bidx.mInDataBlockIdx.x = entry & 3u;
                bidx.mInDataBlockIdx.y = (entry >> 2) & 3u;
                bidx.mInDataBlockIdx.z = (entry >> 4) & 3u;
                return true;
            } else {
                asm volatile("trap;");
            }
        }
        // Block list: surplus thread blocks exit before touching any other memory.
        if (blockIdx.x >= count) {
            bidx.mDataBlockIdx = getInvalidBlockId();
            return false;
        }
        bidx.mDataBlockIdx = mBlockList[1 + blockIdx.x];
    } else {
        bidx.mDataBlockIdx = blockIdx.x + mFirstDataBlockOffset;
    }
    bidx.mInDataBlockIdx.x = threadIdx.x;
    bidx.mInDataBlockIdx.y = threadIdx.y;
    bidx.mInDataBlockIdx.z = threadIdx.z;

    const bool isActive = mActiveMask[bidx.mDataBlockIdx].isActive(bidx.mInDataBlockIdx.x, bidx.mInDataBlockIdx.y, bidx.mInDataBlockIdx.z);

    return isActive;
#else
    NEON_THROW_UNSUPPORTED_OPERATION("Operation supported only on GPU");
#endif
}

template <typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto
bSpan<SBlock>::setAndValidateCPUDevice(Idx&            bidx,
                                       uint32_t const& dataBlockIdx,
                                       uint32_t const& x,
                                       uint32_t const& y,
                                       uint32_t const& z) const -> bool
{

    bidx.mDataBlockIdx = dataBlockIdx;
    bidx.mInDataBlockIdx.x = static_cast<typename Idx::InDataBlockIdx::Integer>(x);
    bidx.mInDataBlockIdx.y = static_cast<typename Idx::InDataBlockIdx::Integer>(y);
    bidx.mInDataBlockIdx.z = static_cast<typename Idx::InDataBlockIdx::Integer>(z);
    const bool isActive = mActiveMask[dataBlockIdx].isActive(bidx.mInDataBlockIdx.x, bidx.mInDataBlockIdx.y, bidx.mInDataBlockIdx.z);
    return isActive;
}

template <typename SBlock>
bSpan<SBlock>::bSpan(typename Idx::DataBlockCount                  firstDataBlockOffset,
                     typename SBlock::BitMask const* NEON_RESTRICT activeMask,
                     Neon::DataView                                dataView)
    : mFirstDataBlockOffset(firstDataBlockOffset),
      mActiveMask(activeMask),
      mDataView(dataView)
{
    std::cout<<"HERE:: SIZE OF "<<sizeof(bSpan<SBlock>)<<std::endl;
}

#if !defined(NEON_WARP_COMPILATION)
template <typename SBlock>
inline void bSpan<SBlock>::getOffsets(size_t* offsets, size_t* length) {
    static std::vector<size_t> cpp_offsets = {
        offsetof(bSpan, mFirstDataBlockOffset),
        offsetof(bSpan, mActiveMask),
        offsetof(bSpan, mDataView)
    };
    
    *length = cpp_offsets.size();
    for (size_t i = 0; i < cpp_offsets.size(); ++i) {
        offsets[i] = cpp_offsets[i];
    }
}
#endif

}  // namespace Neon::domain::details::bGrid

#if !defined(NEON_WARP_COMPILATION)
#pragma GCC diagnostic pop
#endif