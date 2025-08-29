# InfiniteTalk/MultiTalk WAN2.1 RunPod Serverless - Optimized for PyTorch 2.7.1
FROM spxiong/pytorch:2.7.1-py3.10.15-cuda12.6.0-ubuntu22.04

WORKDIR /app

# Environment variables optimized cho CUDA 12.6
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-12.6
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    aria2 ffmpeg wget curl git \
    build-essential python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify PyTorch từ base image
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Install XFormers và FlashAttention tương thích với PyTorch 2.7.1 + CUDA 12.6
RUN pip install --no-cache-dir -U \
    torch==2.7.1 torchvision torchaudio xformers \
    --index-url https://download.pytorch.org/whl/cu126

# Install FlashAttention từ wheel được build sẵn cho PyTorch 2.7
RUN pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Cài đặt SageAttention
RUN pip install --no-cache-dir \
    https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/4b17732a84bc50cf1e8b790e854ee9cd5e2ebfbf/sageattention-2.1.1-cp310-cp310-linux_x86_64.whl

# Core ML/AI packages compatible với PyTorch 2.7.1
RUN pip install --no-cache-dir \
    torchsde==0.2.6 \
    einops==0.8.0 \
    diffusers==0.30.3 \
    transformers==4.45.2 \
    accelerate==1.0.1 \
    omegaconf==2.3.0 \
    safetensors==0.4.5 \
    tqdm==4.66.6 \
    psutil==6.0.0 \
    kornia==0.7.3

# Computer Vision & Media packages
RUN pip install --no-cache-dir \
    av==12.3.0 \
    spandrel==0.3.4 \
    albumentations==1.4.15 \
    onnx==1.16.2 \
    opencv-python==4.10.0.84 \
    color-matcher==0.6.0 \
    segment_anything==1.0 \
    ultralytics==8.2.103 \
    onnxruntime==1.19.2 \
    onnxruntime-gpu==1.19.2 \
    pillow==10.4.0 \
    numpy==1.26.4 \
    imageio==2.36.0 \
    imageio-ffmpeg==0.5.1 \
    soundfile==0.12.1 \
    moviepy==1.0.3

# RunPod & Storage dependencies
RUN pip install --no-cache-dir \
    runpod>=1.7.0 \
    minio>=7.2.0 \
    huggingface-hub==0.25.2 \
    requests>=2.32.0 \
    urllib3>=2.2.0

# Clone ComfyUI với stable version
RUN git clone --branch ComfyUI_v0.3.47 https://github.com/Isi-dev/ComfyUI /app/ComfyUI

# Clone essential custom nodes theo notebook workflow
WORKDIR /app/ComfyUI/custom_nodes
RUN git clone https://github.com/Isi-dev/ComfyUI_WanVideoWrapper && \
    git clone https://github.com/Isi-dev/audio_separation_nodes_comfyui

# Install custom nodes requirements với error handling
RUN cd /app/ComfyUI/custom_nodes/ComfyUI_WanVideoWrapper && \
    (pip install --no-cache-dir -r requirements.txt || echo "⚠️ WanVideoWrapper requirements partially failed") && \
    cd /app/ComfyUI/custom_nodes/audio_separation_nodes_comfyui && \
    (pip install --no-cache-dir -r requirements.txt || echo "⚠️ Audio separation requirements partially failed")

# Create comprehensive model directories
WORKDIR /app
RUN mkdir -p /app/ComfyUI/models/{diffusion_models,text_encoders,vae,clip_vision,loras,transformers} && \
    mkdir -p /app/ComfyUI/{input,output,temp} && \
    mkdir -p /app/output

# ===== OPTIMIZED MODEL DOWNLOAD SECTION =====
# Download với tăng tốc độ và error recovery
ARG ARIA_OPTS="--console-log-level=warn -c -x 16 -s 16 -k 1M --max-tries=3 --retry-wait=5"

# Download core WAN2.1 models
RUN echo "=== Downloading Core WAN2.1 Models ===" && \
    aria2c ${ARIA_OPTS} \
    "https://huggingface.co/Isi99999/Wan_Extras/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    -d /app/ComfyUI/models/text_encoders \
    -o umt5_xxl_fp8_e4m3fn_scaled.safetensors && \
    aria2c ${ARIA_OPTS} \
    "https://huggingface.co/Isi99999/Wan_Extras/resolve/main/wan_2.1_vae.safetensors" \
    -d /app/ComfyUI/models/vae \
    -o wan_2.1_vae.safetensors && \
    aria2c ${ARIA_OPTS} \
    "https://huggingface.co/Isi99999/Wan_Extras/resolve/main/clip_vision_h.safetensors" \
    -d /app/ComfyUI/models/clip_vision \
    -o clip_vision_h.safetensors && \
    echo "✅ Core WAN2.1 models downloaded"

# Download WAN2.1 base model (Q4_K_M quantized)
RUN echo "=== Downloading WAN2.1 Base Model ===" && \
    aria2c ${ARIA_OPTS} \
    "https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q4_K_M.gguf" \
    -d /app/ComfyUI/models/diffusion_models \
    -o wan2.1-i2v-14b-480p-Q4_K_M.gguf && \
    echo "✅ WAN2.1 base model downloaded"

# Download LightX2V speed LoRA
RUN echo "=== Downloading Speed LoRA ===" && \
    aria2c ${ARIA_OPTS} \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" \
    -d /app/ComfyUI/models/loras \
    -o lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors && \
    echo "✅ Speed LoRA downloaded"

# Download MultiTalk model (primary choice)
RUN echo "=== Downloading MultiTalk Model ===" && \
    aria2c ${ARIA_OPTS} \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/WanVideo_2_1_Multitalk_14B_fp8_e4m3fn.safetensors" \
    -d /app/ComfyUI/models/diffusion_models \
    -o WanVideo_2_1_Multitalk_14B_fp8_e4m3fn.safetensors && \
    echo "✅ MultiTalk model downloaded"

# Download InfiniteTalk model (fallback option)
RUN echo "=== Downloading InfiniteTalk Model ===" && \
    aria2c ${ARIA_OPTS} \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp16.safetensors" \
    -d /app/ComfyUI/models/diffusion_models \
    -o Wan2_1-InfiniteTalk-Multi_fp16.safetensors && \
    echo "✅ InfiniteTalk model downloaded"

# Download Wav2Vec2 model cho audio processing
RUN echo "=== Downloading Wav2Vec Model ===" && \
    aria2c ${ARIA_OPTS} \
    "https://huggingface.co/TencentGameMate/chinese-wav2vec2-base/resolve/main/chinese-wav2vec2-base-fairseq-ckpt.pt" \
    -d /app/ComfyUI/models/transformers \
    -o chinese-wav2vec2-base-fairseq-ckpt.pt && \
    echo "✅ Wav2Vec model downloaded"

# COMPREHENSIVE MODEL VERIFICATION
RUN echo "=== COMPREHENSIVE MODEL VERIFICATION ===" && \
    echo "📁 Checking model files existence..." && \
    test -f /app/ComfyUI/models/diffusion_models/wan2.1-i2v-14b-480p-Q4_K_M.gguf && echo "✅ WAN2.1 base model OK" || echo "❌ WAN2.1 base MISSING" && \
    test -f /app/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors && echo "✅ Text encoder OK" || echo "❌ Text encoder MISSING" && \
    test -f /app/ComfyUI/models/vae/wan_2.1_vae.safetensors && echo "✅ VAE OK" || echo "❌ VAE MISSING" && \
    test -f /app/ComfyUI/models/clip_vision/clip_vision_h.safetensors && echo "✅ CLIP Vision OK" || echo "❌ CLIP Vision MISSING" && \
    test -f /app/ComfyUI/models/diffusion_models/WanVideo_2_1_Multitalk_14B_fp8_e4m3fn.safetensors && echo "✅ MultiTalk OK" || echo "❌ MultiTalk MISSING" && \
    test -f /app/ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp16.safetensors && echo "✅ InfiniteTalk OK" || echo "❌ InfiniteTalk MISSING" && \
    test -f /app/ComfyUI/models/transformers/chinese-wav2vec2-base-fairseq-ckpt.pt && echo "✅ Wav2Vec OK" || echo "❌ Wav2Vec MISSING" && \
    test -f /app/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors && echo "✅ Speed LoRA OK" || echo "❌ Speed LoRA MISSING" && \
    echo "📊 Model file sizes:" && \
    du -h /app/ComfyUI/models/diffusion_models/* && \
    echo "🎯 Total model storage:" && \
    du -sh /app/ComfyUI/models/ && \
    echo "=== VERIFICATION COMPLETE ==="

# Copy application handler
COPY wan21_handler.py /app/wan21_handler.py

# Environment variables cho model paths
ENV PYTHONPATH="/app:/app/ComfyUI"
ENV WAN21_MODEL_PATH="/app/ComfyUI/models/diffusion_models/wan2.1-i2v-14b-480p-Q4_K_M.gguf"
ENV WAN21_VAE_PATH="/app/ComfyUI/models/vae/wan_2.1_vae.safetensors"
ENV UMT5_PATH="/app/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
ENV CLIP_VISION_PATH="/app/ComfyUI/models/clip_vision/clip_vision_h.safetensors"
ENV MULTITALK_MODEL_PATH="/app/ComfyUI/models/diffusion_models/WanVideo_2_1_Multitalk_14B_fp8_e4m3fn.safetensors"
ENV INFINITALK_MODEL_PATH="/app/ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp16.safetensors"
ENV W2V_CKPT_PATH="/app/ComfyUI/models/transformers/chinese-wav2vec2-base-fairseq-ckpt.pt"
ENV SPEED_LORA_PATH="/app/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"

# Runtime optimization
ENV OMP_NUM_THREADS=4
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_BACKENDS_CUDNN_BENCHMARK=1

# Final package verification
RUN echo "=== FINAL PACKAGE VERIFICATION ===" && \
    python -c "import torch, torchvision, transformers, diffusers, accelerate; print(f'✅ Core packages OK - PyTorch {torch.__version__}')" && \
    python -c "import runpod, minio; print('✅ RunPod/MinIO OK')" && \
    python -c "import xformers; print(f'✅ XFormers OK - {xformers.__version__}')" || echo "⚠️ XFormers not available" && \
    python -c "import flash_attn; print('✅ FlashAttention OK')" || echo "⚠️ FlashAttention not available" && \
    echo "⚠️ ComfyUI verification skipped (requires GPU runtime)"

# Optimized health check
HEALTHCHECK --interval=30s --timeout=20s --start-period=300s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'🚀 CUDA Ready - {torch.version.cuda}')" && \
        python -c "import os; required_models=['$WAN21_MODEL_PATH','$WAN21_VAE_PATH','$UMT5_PATH','$CLIP_VISION_PATH','$W2V_CKPT_PATH']; missing=[p for p in required_models if not os.path.exists(p)]; assert not missing, f'Missing models: {missing}'; print('🚀 All Models Ready')" || exit 1

# Expose port
EXPOSE 8000

# Run với optimized memory
CMD ["python", "-u", "/app/wan21_handler.py"]





