# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# Multi-stage, multi-arch Containerfile for TensorRT Edge-LLM
# See docs/source/developer_guide/getting-started/container.md for usage.

ARG TRT_VERSION=25.03-py3

# ---------------------------------------------------------------------------
# Stage 1: Python export pipeline (model quantization & ONNX export)
# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/tensorrt:${TRT_VERSION} AS python-export

WORKDIR /workspace/TensorRT-Edge-LLM

# Install Python dependencies first for better layer caching, then copy source.
COPY pyproject.toml requirements.txt ./
COPY tensorrt_edgellm/ tensorrt_edgellm/
RUN pip install --no-cache-dir .

COPY . .

# ---------------------------------------------------------------------------
# Stage 2: C++ build (intermediate — not shipped)
# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/tensorrt:${TRT_VERSION} AS cpp-build

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/TensorRT-Edge-LLM

COPY . .
RUN git submodule update --init --recursive

# Locate the CUDA driver stub (libcuda.so) portably across architectures.
# In NGC containers without a GPU driver, the real libcuda.so is absent and
# in that case we must find the stub that ships with the CUDA toolkit.
RUN CUDA_DRIVER_STUB=$(find -L /usr/local/cuda -name 'libcuda.so' -path '*/stubs/*' 2>/dev/null | head -1) && \
    if [ -z "$CUDA_DRIVER_STUB" ]; then echo "ERROR: libcuda.so stub not found" >&2; exit 1; fi && \
    echo "Found CUDA driver stub: $CUDA_DRIVER_STUB" && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRT_PACKAGE_DIR=/usr \
        -DCUDA_DIR=/usr/local/cuda \
        -DCUDA_DRIVER_LIB="$CUDA_DRIVER_STUB" \
        -DBUILD_UNIT_TESTS=OFF \
    && cmake --build . --parallel "$(nproc)"

# ---------------------------------------------------------------------------
# Stage 3: Lean C++ runtime (no Python, no build tools)
# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/tensorrt:${TRT_VERSION} AS runtime

WORKDIR /workspace

# C++ binaries
COPY --from=cpp-build /workspace/TensorRT-Edge-LLM/build/examples/llm/llm_build \
                      /workspace/TensorRT-Edge-LLM/build/examples/llm/llm_inference \
                      /usr/local/bin/
COPY --from=cpp-build /workspace/TensorRT-Edge-LLM/build/examples/multimodal/visual_build \
                      /usr/local/bin/

# TensorRT plugin
COPY --from=cpp-build /workspace/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so* \
                      /usr/local/lib/

ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}" \
    EDGELLM_PLUGIN_PATH="/usr/local/lib/libNvInfer_edgellm_plugin.so"

ENTRYPOINT ["/bin/bash"]
