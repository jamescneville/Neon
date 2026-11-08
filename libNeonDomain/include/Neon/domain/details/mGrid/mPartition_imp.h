#pragma once

namespace Neon::domain::details::mGrid {

// Count trailing zeros of a non-zero 32-bit value (log2 of a power of two).
NEON_CUDA_HOST_DEVICE inline auto mPartitionCtz(uint32_t v) -> uint32_t
{
#if defined(__CUDA_ARCH__)
    return static_cast<uint32_t>(__ffs(static_cast<int>(v)) - 1);
#elif defined(__GNUC__)
    return static_cast<uint32_t>(__builtin_ctz(v));
#else
    uint32_t n = 0;
    while ((v & 1u) == 0u) {
        v >>= 1;
        ++n;
    }
    return n;
#endif
}

template <typename T, int C, typename SBlock>
mPartition<T, C, SBlock>::mPartition()
    : Neon::domain::details::bGrid::bPartition<T, C, SBlock>(),
      mMemParent(nullptr),
      mMemChild(nullptr),
      mParentBlockID(nullptr),
      mMaskLowerLevel(nullptr),
      mMaskUpperLevel(nullptr),
      mChildBlockID(nullptr),
      mRefFactors(nullptr),
      mSpacing(nullptr)
{
}

template <typename T, int C, typename SBlock>
mPartition<T, C, SBlock>::mPartition(int                         level,
                                     T*                          mem,
                                     T*                          memParent,
                                     T*                          memChild,
                                     int                         cardinality,
                                     typename Idx::DataBlockIdx* neighbourBlocks,
                                     Neon::int32_3d*             origin,
                                     typename Idx::DataBlockIdx* parentBlockID,
                                     MaskT*                      mask,
                                     MaskT*                      maskLowerLevel,
                                     MaskT*                      maskUpperLevel,
                                     typename Idx::DataBlockIdx* childBlockID,
                                     typename Idx::DataBlockIdx* parentNeighbourBlocks,
                                     NghIdx*                     stencilNghIndex,
                                     int*                        refFactors,
                                     int*                        spacing)
    : Neon::domain::details::bGrid::bPartition<T, C, SBlock>(0,
                                                             cardinality,
                                                             mem,
                                                             neighbourBlocks,
                                                             mask,
                                                             origin,
                                                             stencilNghIndex,
                                                             Neon::int32_3d(0, 0, 0) /*TODO@Max this is just a temporary solution*/),
      mLevel(level),
      mMemParent(memParent),
      mMemChild(memChild),
      mParentBlockID(parentBlockID),
      mMaskLowerLevel(maskLowerLevel),
      mMaskUpperLevel(maskUpperLevel),
      mChildBlockID(childBlockID),
      mParentNeighbourBlocks(parentNeighbourBlocks),
      mRefFactors(refFactors),
      mSpacing(spacing)
{
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::getGlobalIndex(Idx gidx) const -> Neon::index_3d
{
    const int sp = (mLevel == 0) ? 1 : mSpacing[mLevel - 1];

    Neon::index_3d ret = this->mOrigin[gidx.getDataBlockIdx()];
    ret.x += gidx.mInDataBlockIdx.x * sp;
    ret.y += gidx.mInDataBlockIdx.y * sp;
    ret.z += gidx.mInDataBlockIdx.z * sp;

    return ret;
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::getLevel() const -> int
{
    return mLevel;
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::printLog([[maybe_unused]] bool masterOnly) const -> void
{
#if defined(NEON_PLACE_CUDA_DEVICE)
    if ((!masterOnly) || (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0)) {
#else
    masterOnly = true;
    if (masterOnly) {
#endif

        printf("mPartition: BEGIN\n");
        Neon::domain::details::bGrid::bPartition<T, C, SBlock>::printLog(masterOnly);
        printf("mPartition: level %d\n", mLevel);
        printf("mPartition: memParent %p\n", mMemParent);
        printf("mPartition: memChild %p\n", mMemChild);
        printf("mPartition: parentBlockID %p\n", mParentBlockID);
        printf("mPartition: maskLowerLevel %p\n", mMaskLowerLevel);
        printf("mPartition: maskUpperLevel %p\n", mMaskUpperLevel);
        printf("mPartition: childBlockID %p\n", mChildBlockID);
        printf("mPartition: parentNeighbourBlocks %p\n", mParentNeighbourBlocks);
        printf("mPartition: parentNeighbourBlocks %p\n", mParentNeighbourBlocks);
        printf("mPartition: END\n");
    }
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::getRefFactor(const int level) const -> int
{
    return mRefFactors[level];
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::getSpacing(const int level) const -> int
{
    return mSpacing[level];
}

template <typename T, int C, typename SBlock>
inline NEON_CUDA_HOST_DEVICE auto mPartition<T, C, SBlock>::childID(const Idx& gidx) const -> uint32_t
{
    const uint32_t blockPitchByCard = SBlock::memBlockSizeX * SBlock::memBlockSizeY * SBlock::memBlockSizeZ;
    const uint32_t inBlockInCardPitch = gidx.mInDataBlockIdx.x +
                                        SBlock::memBlockSizeX * gidx.mInDataBlockIdx.y +
                                        (SBlock::memBlockSizeX * SBlock::memBlockSizeY) * gidx.mInDataBlockIdx.z;
    const uint32_t blockAdnCardPitch = gidx.mDataBlockIdx * blockPitchByCard;
    const uint32_t pitch = blockAdnCardPitch + inBlockInCardPitch;
    return mChildBlockID[pitch];
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::getChild(const Idx& parentCell,
                                                                     NghIdx     child) const -> Idx
{
    Idx childCell;
    childCell.mDataBlockIdx = std::numeric_limits<typename Idx::DataBlockIdx>::max();

    if (hasChildren(parentCell)) {
        childCell.mDataBlockIdx = childID(parentCell);
        int ref = getRefFactor(mLevel);
        childCell.mInDataBlockIdx.x = (ref * parentCell.mInDataBlockIdx.x + child.x) % SBlock::memBlockSizeX;
        childCell.mInDataBlockIdx.y = (ref * parentCell.mInDataBlockIdx.y + child.y) % SBlock::memBlockSizeY;
        childCell.mInDataBlockIdx.z = (ref * parentCell.mInDataBlockIdx.z + child.z) % SBlock::memBlockSizeZ;
    }
    return childCell;
}


template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::childVal(const Idx& childCell,
                                                                     int        card) -> T&
{
    return mMemChild[this->helpGetPitch(childCell, card)];
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::childVal(const Idx& childCell,
                                                                     int        card) const -> const T&
{
    return mMemChild[this->helpGetPitch(childCell, card)];
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::childVal(const Idx&   parentCell,
                                                                     const NghIdx child,
                                                                     int          card,
                                                                     const T&     alternativeVal) const -> NghData
{
    NghData ret;
    ret.mData = alternativeVal;
    ret.mIsValid = false;
    if (!parentCell.isActive() || !hasChildren(parentCell)) {
        return ret;
    }

    Idx childCell = getChild(parentCell, child);

    if (!childCell.isActive()) {
        return ret;
    }

    ret.mIsValid = true;
    ret.mData = childVal(childCell, card);

    return ret;
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::hasChildren(const Idx& cell) const -> bool
{
    if (mMemChild == nullptr || mMaskLowerLevel == nullptr || mLevel == 0) {
        return false;
    }
    // auto chId = childID(cell);
    // auto maxIdx = std::numeric_limits<typename Idx::DataBlockIdx>::max();
    // printf("childID: %x  vs   %x (%x)\n", chId, maxIdx, 4294967295U);
    if (childID(cell) == std::numeric_limits<typename Idx::DataBlockIdx>::max()) {
        return false;
    }
    return true;
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::hasChildren(const Idx& cell, const NghIdx nghDir) const -> bool
{
    if (mMemChild == nullptr || mMaskLowerLevel == nullptr || mLevel == 0) {
        return false;
    }

    Idx nghCell = this->helpGetNghIdx(cell, nghDir);
    if (nghCell.mDataBlockIdx == std::numeric_limits<typename Idx::DataBlockIdx>::max()) {
        return false;
    }
    if (!this->isActive(nghCell)) {
        return false;
    }
    return hasChildren(nghCell);
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::getParent(const Idx& cell) const -> Idx
{
    Idx parentCell;
    parentCell.mDataBlockIdx = std::numeric_limits<typename Idx::DataBlockIdx>::max();
    if (mMemParent != nullptr) {
        parentCell.mDataBlockIdx = mParentBlockID[cell.mDataBlockIdx];
        const Neon::index_3d g = this->getGlobalIndex(cell);
        const uint32_t       sp = getSpacing(mLevel);
        if ((sp & (sp - 1u)) == 0u) {
            // Spacing is a power of two (refinement factor 2): replace the three runtime
            // integer divisions with one shift count and three shifts. Bit-identical to g / sp
            // for non-negative g (which is converted to unsigned exactly as the division does).
            const uint32_t sh = mPartitionCtz(sp);
            parentCell.mInDataBlockIdx.x = (static_cast<uint32_t>(g.x) >> sh) % SBlock::memBlockSizeX;
            parentCell.mInDataBlockIdx.y = (static_cast<uint32_t>(g.y) >> sh) % SBlock::memBlockSizeY;
            parentCell.mInDataBlockIdx.z = (static_cast<uint32_t>(g.z) >> sh) % SBlock::memBlockSizeZ;
        } else {
            parentCell.mInDataBlockIdx.x = (g.x / sp) % SBlock::memBlockSizeX;
            parentCell.mInDataBlockIdx.y = (g.y / sp) % SBlock::memBlockSizeY;
            parentCell.mInDataBlockIdx.z = (g.z / sp) % SBlock::memBlockSizeZ;
        }
    }
    return parentCell;
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::parentVal(const Idx& eId,
                                                                      int        card) -> T&
{
    auto parentCell = getParent(eId);
    if (parentCell.isActive()) {
        return mMemParent[this->helpGetPitch(parentCell, card)];
    }
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::parentVal(const Idx& eId,
                                                                      int        card) const -> const T&
{
    auto parentCell = getParent(eId);
    if (parentCell.isActive()) {
        return mMemParent[this->helpGetPitch(parentCell, card)];
    }
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::hasParent(const Idx& cell) const -> bool
{
    if (!cell.isActive()) {
        return false;
    }
    return getParent(cell).isActive();
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::getUncleFromParent(const Idx&   parent,
                                                                               const NghIdx direction) const -> Idx
{
    // Same as getUncle() but takes an already-computed parent, so callers that need several
    // uncles of the same cell (e.g. restriction deposits over all populations) call getParent once.
    Idx uncle = parent;
    if (uncle.isActive()) {
        uncle = this->helpGetNghIdx(uncle, direction, mParentNeighbourBlocks);
        if (uncle.mDataBlockIdx != std::numeric_limits<typename Idx::DataBlockIdx>::max()) {
            if (!this->isActive(uncle, mMaskUpperLevel)) {
                uncle.mDataBlockIdx = std::numeric_limits<typename Idx::DataBlockIdx>::max();
            }
        }
    }
    return uncle;
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::uncleValAt(const Idx& uncle,
                                                                       int        card) const -> T&
{
    assert(this->isActive(uncle, mMaskUpperLevel));
    return mMemParent[this->helpGetPitch(uncle, card)];
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::getUncle(const Idx&   cell,
                                                                     const NghIdx direction) const -> Idx
{
    Idx uncle = getParent(cell);
    if (uncle.isActive()) {
        uncle = this->helpGetNghIdx(uncle, direction, mParentNeighbourBlocks);
        if (uncle.mDataBlockIdx != std::numeric_limits<typename Idx::DataBlockIdx>::max()) {
            if (!this->isActive(uncle, mMaskUpperLevel)) {
                uncle.mDataBlockIdx = std::numeric_limits<typename Idx::DataBlockIdx>::max();
            }
        }
    }
    return uncle;
}

template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::uncleVal(const Idx&   cell,
                                                                     const NghIdx direction,
                                                                     int          card,
                                                                     const T&     alternativeVal) const -> NghData
{
    NghData ret;
    ret.mData = alternativeVal;
    ret.mIsValid = false;

    Idx uncle = getUncle(cell, direction);
    ret.mIsValid = uncle.isActive();
    if (ret.mIsValid) {
        ret.mData = mMemParent[this->helpGetPitch(uncle, card)];
    }
    return ret;
}


template <typename T, int C, typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C, SBlock>::uncleVal(const Idx&   cell,
                                                                     const NghIdx direction,
                                                                     int          card) const -> T&
{
    Idx uncle = getUncle(cell, direction);
    assert(this->isActive(uncle, mMaskUpperLevel));
    return mMemParent[this->helpGetPitch(uncle, card)];
}

}  // namespace Neon::domain::details::mGrid