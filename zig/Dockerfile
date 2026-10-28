# The engine image (both ranks): TensorFold's native (Zig) engine with the DeepSeek-V4.1-Flash family of BertholomusAI's
# fork, branch deepseek-v41-zig (TF_REPO at TF_REF), plus zig/patches/*.patch if there are any (git apply in the
# checkout's root, filename order). It builds on NVIDIA's PyTorch container, which every Spark already holds: its CUDA
# toolchain compiles the sm_121 fatbins, its NCCL 2.30.7, cuBLAS and libibverbs are what the engine loads at run time,
# its PyTorch makes the kit's RoPE tables, and its Python with Pillow decodes the picture formats other than PNG (the
# engine starts a python3 child for them). No GPU is needed to build.
#
# The two-node branch writes a rank's weight file and Engram's token map only from its Python engine on a first start.
# So a second checkout, the fork's branch deepseek-v41-zig-tp4 at INPUTS_REF, is built only for what it adds, and only
# that is kept: tf-dsv41-rank-cache (the weight file from the checkpoint alone, no GPU, no Python; the same file format,
# which both engines read) and the standalone make_token_map.py and make_rope.py.
# Built by scripts/prepare.sh, never by hand: it passes the build arguments and a build context of this file,
# zig/patches and zig/kit-patches only.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
# Optional ABLIT preparation uses the image's PyTorch and safetensors on CPU.
RUN python3 -m pip install --no-cache-dir safetensors
ARG ZIG_VERSION
ARG ZIG_SHA256
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz && \
    /opt/zig/zig version
ARG SM=121

# the serving engine
ARG TF_REPO
ARG TF_REF
RUN git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${TF_REF}"
COPY zig/patches /opt/tf-patches
RUN cd /tensorfold && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=${SM} -Doptimize=ReleaseFast \
      --prefix /opt/tensorfold -j"$(nproc)" install native && \
    test -x /opt/tensorfold/native/bin/tensorfold-native && test -x /opt/tensorfold/bin/tf-dsv41-lanes && \
    mkdir -p /opt/tensorfold/share/doc/tensorfold && \
    for f in LICENSE NOTICE ATTRIBUTION.md THIRD_PARTY_NOTICES.md LICENSES; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done && \
    if [ -e tools/dsv41-zig/ATTRIBUTION.md ]; then cp -a tools/dsv41-zig/ATTRIBUTION.md /opt/tensorfold/share/doc/tensorfold/DSV41-ATTRIBUTION.md; fi && \
    rm -rf /tensorfold /root/.cache/zig

# the inputs' tools (not the serving engine: nothing of this checkout but the four files below is kept)
ARG INPUTS_REF
RUN git init -q /tf-inputs && cd /tf-inputs && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${INPUTS_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${INPUTS_REF}"
RUN cd /tf-inputs && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=${SM} -Doptimize=ReleaseFast \
      --prefix /opt/inputs -j"$(nproc)" install && \
    test -x /opt/inputs/bin/tf-dsv41-rank-cache && \
    cp /opt/inputs/bin/tf-dsv41-rank-cache /opt/tensorfold/bin/ && \
    mkdir -p /opt/tensorfold/tools/inputs && \
    cp tools/dsv41-zig/inputs/make_token_map.py tools/dsv41-zig/kit/make_rope.py tools/dsv41-zig/kit/make_bias_vl.py \
       /opt/tensorfold/tools/inputs/ && \
    cp LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSE-inputs-branch && \
    rm -rf /tf-inputs /opt/inputs /root/.cache/zig

# the kit's experts cubin rebuilt from source (this recipe): the same fork's branch deepseek-v41-tp2 (its Python engine,
# whose torch extension the kit's cubins/experts_cb.cubin comes from) at TF_PY_REF, with zig/kit-patches/*.patch, compiled
# as that engine compiles it (zig/kit-patches/build_experts.py); kept at /opt/tensorfold/kit-overlay/experts_cb.cubin,
# which the engine reads in place of the kit's when TF_DS_EXPERTS_CUBIN names it (scripts/config.sh)
ARG TF_PY_REF
COPY zig/kit-patches /opt/tf-kit-patches
RUN git init -q /tf-py && cd /tf-py && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_PY_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${TF_PY_REF}" && \
    for p in /opt/tf-kit-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done && \
    python3 /opt/tf-kit-patches/build_experts.py /tf-py/src/tensorfold/cuda/exl3 /tf-kit-build /opt/tensorfold/kit-overlay/experts_cb.cubin && \
    cp LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSE-kit-source-branch && \
    rm -rf /tf-py /tf-kit-build /root/.cache/torch_extensions

# Optional dense-prefill epilogues: preserve the kit's complete original AOT set
# and append eight CPU-compiled, byte-checked sm_121 variants. Default dispatch is
# unchanged; start.sh selects this set only with TF_DS_PF_EP32=1.
ARG KIT_REPO
ARG KIT_REF
RUN git init -q /tf-prefill-kit && cd /tf-prefill-kit && git remote add origin "${KIT_REPO}" && \
    git fetch -q --depth 1 origin "${KIT_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${KIT_REF}"
RUN --network=none mkdir -p /out && cp /opt/tf-kit-patches/gemm_epilogue.py /out/gemm_epilogue.py && \
    NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES=-1 python3 /opt/tf-kit-patches/build_prefill.py \
      /tf-prefill-kit/kit/aot /opt/tensorfold/prefill-aot && \
    if [ -f /tf-prefill-kit/LICENSE ]; then cp /tf-prefill-kit/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSE-prefill-kit; fi && \
    rm -rf /tf-prefill-kit /out/gemm_epilogue.py /out/__pycache__ /root/.triton

ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH} tf.engine=zig tf.ref=${TF_REF} tf.inputs=${INPUTS_REF} tf.kit_source=${TF_PY_REF}
ENV PATH=/opt/tensorfold/native/bin:/opt/tensorfold/bin:${PATH}
WORKDIR /workspace
