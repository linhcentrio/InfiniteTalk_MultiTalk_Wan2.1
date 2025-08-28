#!/bin/bash
# build_wan21_debug.sh - Build WAN2.1 InfiniteTalk/MultiTalk Docker Image
# Optimized for PyTorch 2.7.1 + CUDA 12.6

echo "🚀 Building WAN2.1 InfiniteTalk/MultiTalk với PyTorch 2.7.1 optimization..."

# Set build environment
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

# Build information
BUILD_TAG="wan21-infinitetalk:v2.7.1-optimized"
BUILD_TIME=$(date +"%Y%m%d_%H%M%S")

echo "📦 Build Info:"
echo "   Tag: $BUILD_TAG" 
echo "   Time: $BUILD_TIME"
echo "   Base: spxiong/pytorch:2.7.1-py3.10.15-cuda12.6.0-ubuntu22.04"

# Build với optimized caching và logging
echo "🔨 Starting build process..."
docker build \
    --no-cache \
    --progress=plain \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -t $BUILD_TAG \
    -f Dockerfile . 2>&1 | tee build_${BUILD_TIME}.log

# Check build success
BUILD_STATUS=${PIPESTATUS[0]}
if [ $BUILD_STATUS -ne 0 ]; then
    echo "❌ Build FAILED với exit code $BUILD_STATUS"
    echo "📋 Check build log: build_${BUILD_TIME}.log"
    exit $BUILD_STATUS
fi

echo "✅ Build SUCCESS! Bắt đầu comprehensive verification..."

# ===== COMPREHENSIVE VERIFICATION =====
echo "🔍 === SYSTEM VERIFICATION ==="

# Check PyTorch và CUDA
echo "📊 PyTorch & CUDA Check:"
docker run --rm --gpus all $BUILD_TAG python -c "
import torch
import sys
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')
print(f'✅ CUDA Version: {torch.version.cuda}')
print(f'✅ GPU Count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'✅ GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Check core packages
echo "📦 Core Packages Check:"
docker run --rm $BUILD_TAG python -c "
packages = ['torch', 'torchvision', 'diffusers', 'transformers', 'accelerate', 'xformers', 'runpod', 'minio']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {pkg}: {version}')
    except ImportError:
        print(f'❌ {pkg}: NOT INSTALLED')
"

# Check advanced packages
echo "🔧 Advanced Packages Check:"  
docker run --rm $BUILD_TAG python -c "
import sys
advanced_packages = {
    'flash_attn': 'FlashAttention',
    'sageattention': 'SageAttention', 
    'triton': 'Triton'
}
for pkg, name in advanced_packages.items():
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {name}: {version}')
    except ImportError:
        print(f'⚠️ {name}: Not available (optional)')
"

# Check ComfyUI import
echo "🎨 ComfyUI Integration Check:"
docker run --rm $BUILD_TAG python -c "
import sys
sys.path.insert(0, '/app/ComfyUI')
try:
    import nodes
    print('✅ ComfyUI nodes import: OK')
    import execution
    print('✅ ComfyUI execution import: OK')
except Exception as e:
    print(f'⚠️ ComfyUI import issue: {e}')
"

# ===== MODEL VERIFICATION =====
echo "🤖 === MODEL VERIFICATION ==="
docker run --rm $BUILD_TAG python -c "
import os

# Core models
models = {
    'WAN2.1 Base': '/app/ComfyUI/models/diffusion_models/wan2.1-i2v-14b-480p-Q4_K_M.gguf',
    'Text Encoder': '/app/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors',
    'VAE': '/app/ComfyUI/models/vae/wan_2.1_vae.safetensors',
    'CLIP Vision': '/app/ComfyUI/models/clip_vision/clip_vision_h.safetensors',
    'MultiTalk': '/app/ComfyUI/models/diffusion_models/WanVideo_2_1_Multitalk_14B_fp8_e4m3fn.safetensors',
    'InfiniteTalk': '/app/ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp16.safetensors',
    'Wav2Vec': '/app/ComfyUI/models/transformers/chinese-wav2vec2-base-fairseq-ckpt.pt',
    'Speed LoRA': '/app/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors'
}

total_size = 0
for name, path in models.items():
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024**3  # GB
        total_size += size
        print(f'✅ {name}: {size:.2f} GB')
    else:
        print(f'❌ {name}: MISSING')

print(f'📊 Total model size: {total_size:.2f} GB')
"

# ===== CUSTOM NODES VERIFICATION =====
echo "🔌 === CUSTOM NODES VERIFICATION ==="
docker run --rm $BUILD_TAG python -c "
import os
import sys

# Check custom nodes
custom_nodes = [
    '/app/ComfyUI/custom_nodes/ComfyUI_WanVideoWrapper',
    '/app/ComfyUI/custom_nodes/audio_separation_nodes_comfyui'
]

for node_path in custom_nodes:
    node_name = os.path.basename(node_path)
    if os.path.exists(node_path):
        print(f'✅ {node_name}: Installed')
        # Check if has __init__.py or main files
        files = os.listdir(node_path)
        py_files = [f for f in files if f.endswith('.py')]
        print(f'   📄 Python files: {len(py_files)}')
    else:
        print(f'❌ {node_name}: Missing')
"

# ===== CONTAINER HEALTH TEST =====
echo "🏥 === CONTAINER HEALTH TEST ==="
echo "⏱️ Starting container health test (30s timeout)..."
CONTAINER_ID=$(docker run -d --gpus all $BUILD_TAG)
sleep 10

# Check if container is still running
if docker ps | grep -q $CONTAINER_ID; then
    echo "✅ Container startup: SUCCESS"
    
    # Check health status
    HEALTH_STATUS=$(docker inspect $CONTAINER_ID --format='{{.State.Health.Status}}' 2>/dev/null || echo "no-health-check")
    echo "🏥 Health status: $HEALTH_STATUS"
    
    # Get container logs
    echo "📋 Container logs (last 10 lines):"
    docker logs $CONTAINER_ID --tail 10
else
    echo "❌ Container startup: FAILED"
    echo "📋 Container logs:"
    docker logs $CONTAINER_ID
fi

# Cleanup
docker stop $CONTAINER_ID >/dev/null 2>&1
docker rm $CONTAINER_ID >/dev/null 2>&1

# ===== FINAL SUMMARY =====
echo ""
echo "🎉 === BUILD VERIFICATION COMPLETE ==="
echo "📦 Image: $BUILD_TAG"
echo "📋 Build log: build_${BUILD_TIME}.log"
echo "💾 Image size:"
docker images $BUILD_TAG --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"

echo ""
echo "🚀 Ready for deployment! Usage:"
echo "   docker run --gpus all -p 8000:8000 $BUILD_TAG"
echo ""
echo "📊 Để test RunPod deployment:"
echo "   docker run --gpus all -e RUNPOD_AI_API_KEY=\$YOUR_KEY $BUILD_TAG"
echo ""
echo "✅ Build và verification hoàn tất!"
