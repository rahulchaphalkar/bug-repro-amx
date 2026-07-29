#include <immintrin.h>
#include <intrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#endif

#ifndef XSTATE_MASK_AMX_TILE_CONFIG
#define XSTATE_MASK_AMX_TILE_CONFIG (1ull << 17)
#endif
#ifndef XSTATE_MASK_AMX_TILE_DATA
#define XSTATE_MASK_AMX_TILE_DATA (1ull << 18)
#endif

typedef struct amx_tilecfg {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} amx_tilecfg;

typedef struct feature_report {
    bool amx_tile;
    bool amx_bf16;
    bool amx_int8;
    bool xcr0_tilecfg;
    bool xcr0_tiledata;
    bool os_xstate_api_ok;
    bool os_tilecfg_enabled;
    bool os_tiledata_enabled;
} feature_report;

extern void amx_bf16_kernel(const amx_tilecfg *cfg, const uint16_t *a, const uint16_t *b, float *c);

static feature_report detect_features(void) {
    feature_report r = {0};

    int cpu_info[4] = {0, 0, 0, 0};
    __cpuidex(cpu_info, 7, 0);
    const uint32_t edx = (uint32_t)cpu_info[3];

    r.amx_bf16 = ((edx >> 22) & 1u) != 0;
    r.amx_tile = ((edx >> 24) & 1u) != 0;
    r.amx_int8 = ((edx >> 25) & 1u) != 0;

    const uint64_t xcr0 = (uint64_t)_xgetbv(0);
    r.xcr0_tilecfg = ((xcr0 >> 17) & 1ull) != 0;
    r.xcr0_tiledata = ((xcr0 >> 18) & 1ull) != 0;

#if defined(USE_XSTATE_API) && defined(_WIN32)
    /* Older SDKs may not declare GetEnabledXStateFeatures. */
    HMODULE k32 = GetModuleHandleA("kernel32.dll");
    if (k32) {
        typedef unsigned long long(WINAPI *GetEnabledXStateFeaturesFn)(void);
        GetEnabledXStateFeaturesFn fn = (GetEnabledXStateFeaturesFn)GetProcAddress(k32, "GetEnabledXStateFeatures");
        if (fn) {
            const uint64_t mask = fn();
            r.os_xstate_api_ok = true;
            r.os_tilecfg_enabled = (mask & XSTATE_MASK_AMX_TILE_CONFIG) != 0;
            r.os_tiledata_enabled = (mask & XSTATE_MASK_AMX_TILE_DATA) != 0;
        }
    }
#endif

    return r;
}

static void print_report(const feature_report *r) {
    printf("CPUID.7.0: AMX_TILE=%d AMX_BF16=%d AMX_INT8=%d\n",
           r->amx_tile ? 1 : 0, r->amx_bf16 ? 1 : 0, r->amx_int8 ? 1 : 0);
    printf("XCR0: XTILECFG=%d XTILEDATA=%d\n",
           r->xcr0_tilecfg ? 1 : 0, r->xcr0_tiledata ? 1 : 0);
    if (r->os_xstate_api_ok) {
        printf("OS xstate: XTILECFG=%d XTILEDATA=%d\n",
               r->os_tilecfg_enabled ? 1 : 0, r->os_tiledata_enabled ? 1 : 0);
    } else {
        printf("OS xstate: GetEnabledXStateFeatures unavailable\n");
    }
    fflush(stdout);
}

static bool can_run_amx_bf16(const feature_report *r) {
    const bool cpu_ok = r->amx_tile && r->amx_bf16;
    const bool xcr0_ok = r->xcr0_tilecfg && r->xcr0_tiledata;
    if (!cpu_ok || !xcr0_ok) {
        return false;
    }

    if (r->os_xstate_api_ok) {
        return r->os_tilecfg_enabled && r->os_tiledata_enabled;
    }
    return true;
}

static void init_tilecfg(amx_tilecfg *cfg) {
    *cfg = (amx_tilecfg){0};
    cfg->palette_id = 1;

    cfg->colsb[0] = 32;
    cfg->rows[0] = 16;

    cfg->colsb[1] = 32;
    cfg->rows[1] = 16;

    cfg->colsb[2] = 64;
    cfg->rows[2] = 16;
}

static void init_inputs(uint16_t *a, uint16_t *b) {
    for (int i = 0; i < 16 * 16; ++i) {
        a[i] = 0x3f80; /* bf16(1.0) */
        b[i] = 0x4000; /* bf16(2.0) */
    }
}

static void print_sample(const float *c) {
    printf("C[0..7]:");
    for (int i = 0; i < 8; ++i) {
        printf(" %.3f", c[i]);
    }
    printf("\n");
}

int main(void) {
    feature_report r = detect_features();
    print_report(&r);

    if (!can_run_amx_bf16(&r)) {
        printf("AMX-BF16 not fully enabled on this host. Skipping kernel.\n");
        return 0;
    }

    amx_tilecfg cfg;
    uint16_t a[16 * 16];
    uint16_t b[16 * 16];
    float c[16 * 16] = {0};

    init_tilecfg(&cfg);
    init_inputs(a, b);

    amx_bf16_kernel(&cfg, a, b, c);
    print_sample(c);

    return 0;
}
