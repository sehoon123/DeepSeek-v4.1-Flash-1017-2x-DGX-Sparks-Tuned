# References

What these measurements read or used, and under what license. Nothing listed here is redistributed in `research/`.

## Used by the recipe

The model, its quantization and overlay, TensorFold, BertholomusAI's fork and kit, Mia's AI Lab's recipe and what it
builds on: [`../NOTICE`](../NOTICE) and [`../CREDITS.md`](../CREDITS.md).

## Used for the measurements (not redistributed)

| Tool | Version | License | Use |
| --- | --- | --- | --- |
| [sparkDash](https://github.com/MiaAI-Lab/sparkDash) | local `606e0f3`; upstream `be35be9` compared | Apache-2.0 from 2.0.0, MIT before | the decode bench: `tools/bench/*.mjs` import its prompt picker and streaming client from a local checkout |
| NVIDIA Nsight Systems | 2025.3.2 | NVIDIA's license | rank-1 kernel traces (`nsys`) |
| [perftest](https://github.com/linux-rdma/perftest) (`ib_write_lat`) | 24.01 | BSD or GPL-2.0 | the RDMA wire latency |
| PyTorch, NumPy (in the engine image) | 2.13.0a0, 2.1.0 | BSD-3-Clause | the analysis scripts |

## Read for comparison (nothing copied)

Every GitHub repository the notes or the survey behind `GITHUB-COMPARISON.md` name; the license is what GitHub reports
(`NONE`: no license file, `NOASSERTION`: one GitHub does not identify), the commit is the default branch's on
2026-10-20. Only figures quoted in the notes, with their source, come from them.

| Repository | License | Commit |
| --- | --- | --- |
| [sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000) | MIT | `main` @ `ce69105cb581` |
| [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10) | MIT | `main` @ `081fda975417` |
| [ashhart/Cortheon](https://github.com/ashhart/Cortheon) | MIT | `main` @ `b4b066b46345` |
| [ashhart/TensorFold](https://github.com/ashhart/TensorFold) | Apache-2.0 | `main` @ `d1018b318fb5` |
| [earendil-works/pi](https://github.com/earendil-works/pi) | MIT | `main` @ `26293e74f664` |
| [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) | Apache-2.0 | `main` @ `ef8b26101323` |
| [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold) | Apache-2.0 | `main` @ `17c73e189f5e` |
| [drowzeys/keys-DeepSeek-V4.1-Flash-Abliterated-Mia-2x-Spark-EXL3](https://github.com/drowzeys/keys-DeepSeek-V4.1-Flash-Abliterated-Mia-2x-Spark-EXL3) | NOASSERTION | `main` @ `eb962e67109f` |
| [drowzeys/keys-DeepSeekV4.1-Flash-TR3-Hybrid-1M-Context-8M-KV-Four-DGX-Sparks-with-abliterated-option](https://github.com/drowzeys/keys-DeepSeekV4.1-Flash-TR3-Hybrid-1M-Context-8M-KV-Four-DGX-Sparks-with-abliterated-option) | NOASSERTION | `main` @ `4b69803bf6ec` |
| [Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark) | Apache-2.0 | `main` @ `8ba23807f588` |
| [Elevatormusic/TensorFold](https://github.com/Elevatormusic/TensorFold) | Apache-2.0 | `main` @ `67f983f24bfa` |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | Apache-2.0 | `main` @ `8dc1163fbfbe` |
| [jayleaton/deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark) | Apache-2.0 | `main` @ `853310dab967` |
| [knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4) | AGPL-3.0 | `main` @ `403a5aa2de35` |
| [knapcio/speedbench](https://github.com/knapcio/speedbench) | Apache-2.0 | `main` @ `e95a638b5f92` |
| [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) | Apache-2.0 | `master` @ `2cc7f66aaef7` |
| [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) | MIT | `main` @ `a8b463671006` |
| [MiaAI-Lab/DeepSeek-V4-Flash-Dual-DGX-Spark-1M-Context](https://github.com/MiaAI-Lab/DeepSeek-V4-Flash-Dual-DGX-Spark-1M-Context) | NONE | `main` @ `d84319952b41` |
| [MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark) | MIT | `main` @ `fdcd538fbf95` |
| [MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks) | Apache-2.0 | `main` @ `31eea4608f8b` |
| [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) | AGPL-3.0 | `main` @ `64b02dea8681` |
| [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) | AGPL-3.0 | `main` @ `6f7d1590ad49` |
| [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) | Apache-2.0 | `main` @ `3fd60dfc836f` |
| [MiaAI-Lab/sparkDash](https://github.com/MiaAI-Lab/sparkDash) | Apache-2.0 | `main` @ `be35be9f594c` |
| [ResolveIO/deepseek-v4.1-1017-tensorfold](https://github.com/ResolveIO/deepseek-v4.1-1017-tensorfold) | Apache-2.0 | `main` @ `64e90cd9b952` |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | Apache-2.0 | `main` @ `07eb671b6af0` |
| [tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark) | MIT | `main` @ `43201d1c2f28` |
| [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark) | MIT | `main` @ `9fa230d3eb11` |
| [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | MIT | `master` @ `a3bfbfd0d3c0` |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | Apache-2.0 | `main` @ `b8e9001bf956` |
| [Zeuss5/cuda-exl3](https://github.com/Zeuss5/cuda-exl3) | NOASSERTION | `master` @ `6a1ffc34866e` |

## Ideas referred to

- Speculative decoding: Leviathan, Kalman and Matias, "Fast Inference from Transformers via Speculative Decoding" (ICML 2023);
  Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023).
- Prompt lookup decoding (Apoorv Saxena, 2023): checking text copied from the context in one pass, which copy windows widen.
- Platt scaling (John Platt, 1999) and Dinkelbach's method for ratio objectives (Werner Dinkelbach, 1967): tried on the
  drafter's confidence and the depth rule, not adopted.
