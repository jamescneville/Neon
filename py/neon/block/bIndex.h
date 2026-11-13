#pragma once

#include "Neon/domain/details/bGrid/bIndex.h"

// TODO: currently, all types and builtins need to be in the wp:: namespace
namespace wp
{

// import types into this namespace
using NeonBlockIdx = ::Neon::domain::details::bGrid::bIndex<Neon::domain::details::bGrid::BlockDefault>;

// Setup-time helper that flags the voxel `a` in a windowed flag array (4x4x4 data blocks). The array address is passed
// as three 16-bit chunks (Warp int literals are 32-bit). Layout, in bytes:
//   [0, 4)   int: first data block of the window (written by the host before each launch)
//   [8]      set when a flagged voxel lies beyond the window, i.e. another window is needed
//   [16, ..) flags for classes 0..2, class-major: flag[16 + cls * nblk * 64 + (block - first) * 64 + x + 4y + 16z]
// nblk is the window size in data blocks. Writes of the same value from many threads are benign.
CUDA_CALLABLE inline auto neon_mark_voxel(const NeonBlockIdx& a, int p0, int p1, int p2, int cls, int nblk) -> void
{
    unsigned long long addr = (static_cast<unsigned long long>(static_cast<unsigned int>(p2)) << 32) |
                              (static_cast<unsigned long long>(static_cast<unsigned int>(p1)) << 16) |
                              static_cast<unsigned long long>(static_cast<unsigned int>(p0));
    unsigned char*  flags = reinterpret_cast<unsigned char*>(addr);
    const long long rel = static_cast<long long>(a.getDataBlockIdx()) - static_cast<long long>(*reinterpret_cast<const int*>(flags));
    if (rel < 0) {
        return;
    }
    if (rel >= nblk) {
        flags[8] = 1;
        return;
    }
    const long long local = static_cast<long long>(a.getInDataBlockIdx().x) + 4 * static_cast<long long>(a.getInDataBlockIdx().y) +
                            16 * static_cast<long long>(a.getInDataBlockIdx().z);
    flags[16 + (static_cast<long long>(cls) * nblk + rel) * 64 + local] = 1;
}

CUDA_CALLABLE inline auto neon_print(const NeonBlockIdx& a) -> void
{
    printf("neon_print - NeonBlockIdx(%d, %d, %d, %d)\n", a.getDataBlockIdx(),a.getInDataBlockIdx().x,   a.getInDataBlockIdx().y, a.getInDataBlockIdx().z);
}
}
