# Experimental gfx90a GDN prefill port: rejected by numerical gate

This directory is research, not the install path. Do not enable this candidate for model inference or deployment. Hardware validation rejected it at the unchanged reference threshold.

- Base: [llama.cpp 7592b61](https://github.com/ggml-org/llama.cpp/commit/7592b61d3b53ca5b39a25a03508fa1246b78c579). The pinned HIP GDN MMA path requires RDNA3.5 and at least 2,048 tokens; its compiled WMMA body also excludes CDNA. An MI210 256-token microbatch falls to the existing recurrent GPU kernel, not to CPU.
- Port: [BLSharda #26001](https://github.com/ggml-org/llama.cpp/pull/26001) and [davetha patch 13, 5ff86324](https://github.com/davetha/llama.cpp-mi210/blob/5ff86324d7999932227a58586b60951c843e7880/patches/13-gdn-chunked-cdna.patch), MIT; source notice retained. The adaptation uses wave64 and MFMA-native lane layout. Unlike the fork, it routes only opt-in gfx90a, 256-token single sequence, 128-wide F32, scalar gate, final state only, strict tensor dimensions/strides. Dispatch, temporary pool graph bypass and state-tail fusion skip share one predicate. Decode, partial tails and all unsupported shapes stay with upstream.
- Scope: exact compiled ISA objects on ROCm 7.2.4, `--offload-arch=gfx90a`. Verified the device object contains the three kernels, 33 `v_mfma_f32_16x16x16f16` instructions, no trap instructions; isolated library still linked RCCL. No weights or deployment changes.
- Actual hardware operator comparison against GGML CPU FP32 using deterministic 8 Q/K and 24 V heads, 128-wide, 256 tokens, strided V with sentinel-filled skipped QKV lanes. Recurrent control output/state NMSE was ~1.3–1.65e-14 across fresh, continued and reset. Candidate fresh attention/output NMSE **2.27662e-6** (fails **2e-7**), with final state **3.18615e-8**. Continued/reset were computed but not scored by the fail-fast test. This is not model accuracy or speed evidence and does not explain long cold first-token latency.

## Rebuild / CPU-only references

In a disposable checkout of the exact base, run `git apply --check` against the patch, then apply it. Use a separate HIP build (ROCm and compiler versions above) and target only gfx90a. The host-side reference checks the dispatch guard and the mathematical recurrence/chunk equivalence for fresh, continued and reset states; it does not validate HIP instructions.

```sh
c++ -std=c++17 -O2 -I/path/to/llama.cpp/ggml/src/ggml-cuda \
  gfx90a-gdn-reference.cpp -o ./gfx90a-gdn-reference
./gfx90a-gdn-reference
```

The separate `gfx90a-gdn-device.cpp` links GGML (`-lggml -lggml-base`) and accepts an explicit backend name, Q/K head count and V head count. Compare the recurrent control and candidate only on a dedicated authorized device. Do not increase the threshold. No hypothesis about production latency follows from this test.

## Precision follow-up: source compiled, device validation pending

[AMD MI200 ISA, Table 25](https://developer.amd.com/wp-content/resources/CDNA2_Shader_ISA_18November2021.pdf) specifies that the actual F16 MFMA op flushes denormals. The CPU-only `localize-rounding.cpp` samples 24 synthetic head/value rows at the same fixture geometry. Ordinary half conversion gives first-fixture output NMSE 6.26e-8; flushing converted-half operands gives 2.59e-6 with state 3.09e-8, close to the device's 2.28e-6 output and 3.19e-8 state. The intra-chunk QK × innovation contributes 2.21e-6 alone in the flush model. This is evidence for the error mechanism, not a device measurement of the correction.

`0002-gfx90a-qk-half-rescale-experiment.patch` (apply after the first patch using `git apply --check`) multiplies only the small Q/QK operands by 256 before half conversion, then applies the exact inverse shift in the FP32 outputs. It preserves the state update and the existing dispatch/threshold. In the flush model, the two synthetic fixtures' output NMSE falls to 6.39e-8 and 7.63e-8. The exact follow-up object and HIP library compiled and linked for gfx90a/ROCm 7.2.4, without running its kernels. Keep it disabled until the unchanged device numerical gate passes with fresh, continued and reset state; model-level performance remains unmeasured.

```sh
clang++ -O2 -std=c++17 -Wall -Wextra -Werror localize-rounding.cpp -o ./localize-rounding
./localize-rounding
```
