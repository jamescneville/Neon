#pragma once

#include "Neon/domain/details/bGrid/bIndex.h"

// TODO: currently, all types and builtins need to be in the wp:: namespace
namespace wp
{

// import types into this namespace
using NeonBlockIdx = ::Neon::domain::details::bGrid::bIndex<Neon::domain::details::bGrid::BlockDefault>;

// Setup-time helper: flag[4*block + cls] = 1 for the data block holding `a`. The flag array address is passed as
// three 16-bit chunks (Warp int literals are 32-bit). Writes of the same value from many threads are benign.
CUDA_CALLABLE inline auto neon_mark_block(const NeonBlockIdx& a, int p0, int p1, int p2, int cls, int nmax) -> void
{
    unsigned long long addr = (static_cast<unsigned long long>(static_cast<unsigned int>(p2)) << 32) |
                              (static_cast<unsigned long long>(static_cast<unsigned int>(p1)) << 16) |
                              static_cast<unsigned long long>(static_cast<unsigned int>(p0));
    unsigned char* flags = reinterpret_cast<unsigned char*>(addr);
    const int      b = static_cast<int>(a.getDataBlockIdx());
    if (b >= nmax) {
        flags[4 * nmax] = 1;  // overflow marker
        return;
    }
    flags[4 * b + cls] = 1;
}

// Voxel-granular variant: flag[cls * nvox + block * 64 + (x + 4y + 16z)] = 1 (4x4x4 data blocks).
CUDA_CALLABLE inline auto neon_mark_voxel(const NeonBlockIdx& a, int p0, int p1, int p2, int cls, int nvox) -> void
{
    unsigned long long addr = (static_cast<unsigned long long>(static_cast<unsigned int>(p2)) << 32) |
                              (static_cast<unsigned long long>(static_cast<unsigned int>(p1)) << 16) |
                              static_cast<unsigned long long>(static_cast<unsigned int>(p0));
    unsigned char* flags = reinterpret_cast<unsigned char*>(addr);
    const long long local = static_cast<long long>(a.getInDataBlockIdx().x) + 4 * static_cast<long long>(a.getInDataBlockIdx().y) +
                            16 * static_cast<long long>(a.getInDataBlockIdx().z);
    const long long flat = static_cast<long long>(a.getDataBlockIdx()) * 64 + local;
    if (flat >= nvox) {
        flags[3LL * nvox] = 1;  // overflow marker
        return;
    }
    flags[static_cast<long long>(cls) * nvox + flat] = 1;
}

CUDA_CALLABLE inline auto neon_print(const NeonBlockIdx& a) -> void
{
    printf("neon_print - NeonBlockIdx(%d, %d, %d, %d)\n", a.getDataBlockIdx(),a.getInDataBlockIdx().x,   a.getInDataBlockIdx().y, a.getInDataBlockIdx().z);
}
}
