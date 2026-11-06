#pragma once

#ifndef SYRIS_WARP_TILE_WIDTH
#define SYRIS_WARP_TILE_WIDTH 0
#endif

static_assert(SYRIS_WARP_TILE_WIDTH == 0 || SYRIS_WARP_TILE_WIDTH == 1 ||
              SYRIS_WARP_TILE_WIDTH == 2 || SYRIS_WARP_TILE_WIDTH == 4 ||
              SYRIS_WARP_TILE_WIDTH == 8 || SYRIS_WARP_TILE_WIDTH == 16 ||
              SYRIS_WARP_TILE_WIDTH == 32, "Warp tile width must be 0 or a divisor of 32");

// A true return with index == pixel_count is padding in a partial edge tile.
// Those lanes must continue to the next shuffle, not exit the kernel early.
__device__ __forceinline__ bool nextProjectionPixel(unsigned *counter, uint2 size,
                                                    unsigned &index)
{
    const unsigned pixel_count = size.x * size.y;
#if SYRIS_WARP_TILE_WIDTH > 0
    // The public launcher uses full 1D warps. Keep custom partial-warp and
    // multidimensional launches correct using the original pixel scheduler.
    if (blockDim.x % 32 == 0 && blockDim.y == 1 && blockDim.z == 1) {
        constexpr unsigned width = SYRIS_WARP_TILE_WIDTH;
        constexpr unsigned height = 32 / width;
        const unsigned columns = (size.x + width - 1) / width;
        const unsigned rows = (size.y + height - 1) / height;
        const unsigned lane = threadIdx.x % 32;
        unsigned tile = 0;
        if (lane == 0) tile = atomicAdd(counter, 1);
        tile = __shfl_sync(0xffffffffu, tile, 0);
        if (tile >= columns * rows) return false;
        const unsigned col = (tile % columns) * width + lane % width;
        const unsigned row = (tile / columns) * height + lane / width;
        index = col < size.x && row < size.y ? row * size.x + col : pixel_count;
        return true;
    }
#endif
    index = atomicAdd(counter, 1);
    return index < pixel_count;
}
