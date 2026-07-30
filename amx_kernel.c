#include <immintrin.h>
#include <stdint.h>

struct amx_tilecfg;

/* AMX BF16 kernel using compiler intrinsics instead of hand-written asm.
   Same signature and behavior as the .S version:
     cfg = tile configuration
     a, b = bf16 input matrices (16x16, stride 32 bytes)
     c = float32 output (stride 64 bytes)
   Intrinsics emit the exact same tile instructions, so this only removes
   hand-encoding as a variable when diagnosing the #UD crash. */
void amx_bf16_kernel(const struct amx_tilecfg *cfg, const uint16_t *a,
                     const uint16_t *b, float *c) {
    _tile_loadconfig(cfg);
    _tile_loadd(0, a, 32);
    _tile_loadd(1, b, 32);
    _tile_zero(2);
    _tile_dpbf16ps(2, 0, 1);
    _tile_stored(2, c, 64);
    _tile_release();
}
