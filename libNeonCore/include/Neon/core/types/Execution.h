#pragma once
#if !defined(NEON_WARP_COMPILATION)
#include <array>
#include <iostream>
#include <string>
#include <vector>
#endif

#include "Neon/core/types/DataUse.h"

namespace Neon {

enum struct Execution
{
    device = 0 /**< A device accelerated execution */,
    host = 1 /**< Pre/post processing execution done on the host */
};

struct ExecutionUtils
{
    constexpr static int numConfigurations = 2;

    /**
     * Safely convert a ExecutionType to string
     *
     * @param option
     * @return
     */
    static auto toString(Neon::Execution option) -> const char*;

    /**
     * Safely convert an Execution to integer
     * @param option
     * @return
     */
    static auto toInt(Neon::Execution option) -> int;
#if !defined(NEON_WARP_COMPILATION)

    static auto getAllOptions()
        -> const std::array<Execution, ExecutionUtils::numConfigurations>&;
#endif
    static auto getCompatibleOptions(Neon::DataUse dataUse)
        -> std::vector<Execution>;

    static auto checkCompatibility(Neon::DataUse   dataUse,
                                   Neon::Execution execution)
        -> bool;

    static auto fromInt(int val) -> Execution;

   private:
#if !defined(NEON_WARP_COMPILATION)
    static constexpr std::array<Execution, ExecutionUtils::numConfigurations> mAllOptions{Execution::device, Execution::host};
#endif
};

/**
 * operator<<
 *
 * @param os
 * @param m
 * @return
 */
std::ostream& operator<<(std::ostream& os, Neon::Execution const& m);
}  // namespace Neon
