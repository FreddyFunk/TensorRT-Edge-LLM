<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.
-->

# Container Images

TensorRT Edge-LLM ships a multi-stage, multi-arch `Containerfile` that produces
two image targets:

| Target | Contents | Use case |
|:-------|:---------|:---------|
| `python-export` | Python export pipeline (quantization & ONNX export) | Model preparation |
| *(default)* `runtime` | Lean C++ binaries only (no Python, no build tools) | Engine build + inference on x86 or aarch64 |

The images are based on the
[NGC TensorRT container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt)
and support both **x86_64** (desktop/server GPU) and **aarch64** (Jetson / DRIVE).

---

## Building the Images

> **Note:** `--target` selects which *build stage* to stop at.
> `-t` (short for `--tag`) assigns a *name* to the resulting image.
> Without `--target`, the build runs all stages and produces the final
> `runtime` image.

No GPU is required to **build** the images. The C++ stage links against a CUDA
driver stub and the Python stage only installs packages. This makes it possible
to build in CI pipelines or on headless servers, then deploy the images to
GPU-equipped machines for inference.

### C++ runtime (default target)

```bash
podman build -t tensorrt-edgellm .
```

### Python export pipeline

```bash
podman build --target python-export -t tensorrt-edgellm-export .
```

### Cross-build for aarch64 on an x86_64 host

This requires `qemu-user-static` to be installed on the host.

```bash
podman build --platform linux/arm64 -t tensorrt-edgellm-arm64 .
```

### Build arguments

| Argument | Description | Default |
|:---------|:------------|:--------|
| `TRT_VERSION` | NGC TensorRT image tag | `25.03-py3` |

Example:

```bash
podman build --build-arg TRT_VERSION=25.06-py3 -t tensorrt-edgellm .
```

---

## Example: Export a Model

The `python-export` image comes with the full Python export pipeline
pre-installed. A typical workflow quantizes a HuggingFace model and exports it
to ONNX, producing files that can be compiled into TensorRT engines on the edge
device.

```bash
# Launch the export container with GPU access and a shared output directory
podman run --rm -it --device nvidia.com/gpu=all \
  -v ./workspace:/workspace/output tensorrt-edgellm-export

# --- inside the container ---

# 1. Quantize (optional, skip for FP16/BF16)
tensorrt-edgellm-quantize-llm \
  --model_dir Qwen/Qwen3-0.6B \
  --quantization fp8 \
  --output_dir /workspace/output/quantized/

# 2. Export to ONNX
tensorrt-edgellm-export-llm \
  --model_dir /workspace/output/quantized/ \
  --output_dir /workspace/output/onnx/
```

The output directory will contain (example for Qwen3-0.6B with FP8):

```
workspace/
├── quantized/
│   ├── model.safetensors
│   ├── modelopt_state.pth
│   ├── config.json
│   ├── generation_config.json
│   ├── hf_quant_config.json
│   ├── chat_template.jinja
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── added_tokens.json
│   ├── merges.txt
│   └── vocab.json
└── onnx/
    ├── model.onnx
    ├── onnx_model.data
    ├── config.json
    ├── embedding.safetensors
    ├── processed_chat_template.json
    ├── chat_template.jinja
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── added_tokens.json
    ├── merges.txt
    └── vocab.json
```

> **Note:** The exact set of tokenizer files varies by model. For example,
> SentencePiece-based models (e.g. Llama) produce a `tokenizer.model` file
> instead of `merges.txt` and `vocab.json`.

Transfer the `onnx/` directory to the edge device, then use the `runtime` image
(or a native build) to compile it into TensorRT engines and run inference.

### Available export tools

| Command | Purpose |
|:--------|:--------|
| `tensorrt-edgellm-quantize-llm` | Quantize a model (FP8, INT4 AWQ, NVFP4, MXFP8, INT8 SQ) |
| `tensorrt-edgellm-export-llm` | Export LLM to ONNX |
| `tensorrt-edgellm-quantize-draft` | Quantize a speculative decoding draft model |
| `tensorrt-edgellm-export-draft` | Export draft model to ONNX |
| `tensorrt-edgellm-export-visual` | Export visual encoder (for VLMs) |
| `tensorrt-edgellm-export-audio` | Export audio model |
| `tensorrt-edgellm-insert-lora` | Add dynamic LoRA support to an ONNX model |
| `tensorrt-edgellm-process-lora` | Process LoRA adapter weights |
| `tensorrt-edgellm-merge-lora` | Merge LoRA weights into a base model |

For detailed usage of each tool, run the command with `--help` or see the
[Quick Start Guide](quick-start-guide.md) and [Examples](examples.md).

---

## Run with GPU

### Prerequisites

GPU access inside a container requires the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
with
[CDI support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html).

After installing the toolkit, generate the CDI spec (required once, and after
driver updates):

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# Verify that the GPU appears
nvidia-ctk cdi list
```

### Podman (CDI)

```bash
podman run --rm -it --device nvidia.com/gpu=all \
  -v ./workspace:/workspace/output tensorrt-edgellm
```

### Docker

```bash
docker run --rm -it --gpus all \
  -v ./workspace:/workspace/output tensorrt-edgellm
```

---

## Example: Build Engine and Run Inference

After exporting a model to ONNX (see [above](#example-export-a-model)), launch
the runtime container and build a TensorRT engine, then run inference.

```bash
# Launch the runtime container with GPU access
podman run --rm -it --device nvidia.com/gpu=all \
  -v ./workspace:/workspace/output tensorrt-edgellm

# --- inside the container ---

# 1. Build the TensorRT engine from the ONNX export
llm_build --onnxDir /workspace/output/onnx --engineDir /workspace/output/engine

# 2. Create a test input
cat > /tmp/input.json << 'EOF'
{
    "max_generate_length": 128,
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "What is TensorRT?"}
            ]
        }
    ]
}
EOF

# 3. Run inference
llm_inference --engineDir /workspace/output/engine \
  --inputFile /tmp/input.json \
  --outputFile /tmp/output.json \
  --dumpOutput
```

For the full input JSON format (multi-turn, multimodal, LoRA, etc.), see the
[Input Format](input-format.md) documentation.
