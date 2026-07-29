.text
.intel_syntax noprefix

.globl amx_bf16_kernel
.p2align 4, 0x90
amx_bf16_kernel:
    /* Windows x64 ABI: rcx=cfg, rdx=a, r8=b, r9=c */
    mov r10, 32
    mov r11, 64

    ldtilecfg [rcx]
    tileloadd tmm0, [rdx + r10]
    tileloadd tmm1, [r8 + r10]
    tilezero tmm2
    tdpbf16ps tmm2, tmm0, tmm1
    tilestored [r9 + r11], tmm2
    tilerelease
    ret
