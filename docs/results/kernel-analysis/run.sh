#!/bin/sh
# Rebuild and rerun the kernel timing harnesses. Run from the repository root with the pinned
# KleidiAI checkout in vendor/kleidiai-v1.32.0. Outputs replace the .txt files in this folder;
# conditions.txt records power source and load averages before and after.
set -eu
here=docs/results/kernel-analysis
kai=vendor/kleidiai-v1.32.0
kd=$kai/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp
bin=$(mktemp -d)
flags="-O3 -march=armv8.2-a+dotprod+fp16 -I $kai"
clang -O2 -march=armv8.2-a+dotprod "$here/sdot_latency.c" -o "$bin/sdot_latency"
clang -O2 -march=armv8.2-a+dotprod "$here/sdot_peak.c" -o "$bin/sdot_peak"
clang $flags "$here/gemv_bench.c" "$kd/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.c" \
  "$kd/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod_asm.S" -o "$bin/gemv_bench"
clang $flags "$here/gemv_1x8_bench.c" "$kd/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.c" \
  "$kd/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod_asm.S" \
  "$kd/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.c" -o "$bin/gemv_1x8_bench"
clang $flags "$here/gemm_bench.c" "$kd/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.c" -o "$bin/gemm_bench"
conditions() { echo "$1: $(pmset -g batt | head -1 | sed 's/^Now drawing from //'); load averages $(sysctl -n vm.loadavg)"; }
conditions before > "$here/conditions.txt"
for t in sdot_latency sdot_peak gemv_bench gemv_1x8_bench; do "$bin/$t" > "$here/$t.txt"; done
"$bin/gemm_bench" 4 > "$here/gemm_bench.txt"
conditions after >> "$here/conditions.txt"
rm -r "$bin"
