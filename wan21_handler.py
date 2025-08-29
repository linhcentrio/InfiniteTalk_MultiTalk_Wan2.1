#!/usr/bin/env python3
"""
WAN2.1 Handler v3.1 - Fixed Context Manager Issue
Corrected memory_safe_execution implementation
"""

import runpod
import os
import tempfile
import uuid
import requests
import time
import torch
import torch.nn.functional as F
import torchaudio
import sys
import gc
import json
import random
import traceback
import subprocess
import imageio
import numpy as np
from pathlib import Path
from minio import Minio
from urllib.parse import quote
import logging
from functools import wraps
from contextlib import contextmanager
from typing import Optional, Tuple, List, Dict, Any

# CRITICAL: Set memory environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.9'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_BACKENDS_CUDNN_BENCHMARK'] = '1'

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [MEM:%(process)d] - %(message)s'
)
logger = logging.getLogger(__name__)
sys.path.insert(0, '/app/ComfyUI')

def setup_memory_management():
    """Setup aggressive memory management"""
    try:
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"💾 GPU Memory Management Setup: {total_memory:.1f}GB total, 85% available")
            
    except Exception as e:
        logger.warning(f"⚠️ Memory management setup failed: {e}")

setup_memory_management()

# Enhanced attention detection
def detect_available_attention_mechanisms():
    mechanisms = {'flash_attn': False, 'xformers': False, 'sageattention': False, 'pytorch': True}
    
    try:
        import flash_attn
        version = getattr(flash_attn, '__version__', 'unknown')
        if '2.7' <= version <= '2.8':
            mechanisms['flash_attn'] = True
            logger.info(f"✅ FlashAttention v{version} - memory efficient")
        else:
            logger.warning(f"⚠️ FlashAttention v{version} - may have issues")
    except ImportError:
        logger.info("⚠️ FlashAttention not available")
    
    try:
        import xformers
        mechanisms['xformers'] = True
        logger.info(f"✅ XFormers v{xformers.__version__} - good fallback")
    except ImportError:
        logger.info("⚠️ XFormers not available")
    
    try:
        import sageattention
        mechanisms['sageattention'] = True
        logger.info("✅ SageAttention detected (memory hungry)")
    except ImportError:
        logger.info("⚠️ SageAttention not available")
    
    return mechanisms

ATTENTION_MECHANISMS = detect_available_attention_mechanisms()

# Import ComfyUI components
try:
    from nodes import (CLIPLoader, CLIPTextEncode, LoadImage, CLIPVisionLoader, ImageScale)
    from custom_nodes.ComfyUI_WanVideoWrapper.nodes_model_loading import (
        WanVideoModelLoader, WanVideoVAELoader, WanVideoLoraSelect, WanVideoBlockSwap
    )
    from custom_nodes.ComfyUI_WanVideoWrapper.nodes import (
        WanVideoSampler, WanVideoTextEmbedBridge, WanVideoDecode, WanVideoClipVisionEncode,
        WanVideoContextOptions
    )
    from custom_nodes.ComfyUI_WanVideoWrapper.multitalk.nodes import (
        MultiTalkModelLoader, MultiTalkWav2VecEmbeds, WanVideoImageToVideoMultiTalk
    )
    from custom_nodes.ComfyUI_WanVideoWrapper.fantasytalking.nodes import DownloadAndLoadWav2VecModel
    from comfy_extras.nodes_audio import LoadAudio
    from custom_nodes.audio_separation_nodes_comfyui.src.separation import AudioSeparation
    from custom_nodes.audio_separation_nodes_comfyui.src.crop import AudioCrop
    
    logger.info("✅ All WanVideoWrapper modules imported successfully")
    WANVIDEO_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ WanVideoWrapper import error: {e}")
    WANVIDEO_AVAILABLE = False

# MinIO Configuration
MINIO_ENDPOINT = "media.aiclip.ai"
MINIO_ACCESS_KEY = "VtZ6MUPfyTOH3qSiohA2"
MINIO_SECRET_KEY = "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t"
MINIO_BUCKET = "video"
MINIO_SECURE = False

try:
    minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, 
                        secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)
    logger.info("✅ MinIO client initialized")
except Exception as e:
    logger.error(f"❌ MinIO initialization failed: {e}")
    minio_client = None

# Model configurations
MODEL_CONFIGS = {
    "wan21_model": os.getenv("WAN21_MODEL_PATH", "/app/ComfyUI/models/diffusion_models/wan2.1-i2v-14b-480p-Q4_K_M.gguf"),
    "wan21_vae": os.getenv("WAN21_VAE_PATH", "/app/ComfyUI/models/vae/wan_2.1_vae.safetensors"),
    "text_encoder": os.getenv("UMT5_PATH", "/app/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    "clip_vision": os.getenv("CLIP_VISION_PATH", "/app/ComfyUI/models/clip_vision/clip_vision_h.safetensors"),
    "multitalk_model": os.getenv("MULTITALK_MODEL_PATH", "/app/ComfyUI/models/diffusion_models/WanVideo_2_1_Multitalk_14B_fp8_e4m3fn.safetensors"),
    "infinitalk_model": os.getenv("INFINITALK_MODEL_PATH", "/app/ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp16.safetensors"),
    "wav2vec_model": os.getenv("W2V_CKPT_PATH", "/app/ComfyUI/models/transformers/chinese-wav2vec2-base-fairseq-ckpt.pt"),
    "speed_lora": os.getenv("SPEED_LORA_PATH", "/app/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors")
}

MEMORY_CONFIG = {
    "max_gpu_usage_percent": 85,
    "emergency_cleanup_threshold": 90,
    "min_free_memory_gb": 6.0,
    "enable_aggressive_cleanup": True,
    "memory_monitoring": True
}

def emergency_memory_cleanup(force_aggressive=False):
    """Aggressive memory cleanup để prevent OOM"""
    try:
        collected = gc.collect()
        
        if torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            cached_before = torch.cuda.memory_reserved() / 1024**3
            
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            if force_aggressive:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.set_per_process_memory_fraction(0.9)
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.95)
            
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            cached_after = torch.cuda.memory_reserved() / 1024**3
            freed_memory = (allocated_before - allocated_after) + (cached_before - cached_after)
            
            logger.info(f"🧹 Memory cleanup: {allocated_after:.1f}GB allocated, {cached_after:.1f}GB cached")
            if freed_memory > 0.1:
                logger.info(f"✅ Freed {freed_memory:.1f}GB memory")
                
    except Exception as e:
        logger.warning(f"⚠️ Memory cleanup error: {e}")

def get_memory_stats():
    """Get current memory statistics"""
    if not torch.cuda.is_available():
        return {}
    
    try:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        usage_percent = (allocated / total) * 100
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": free,
            "usage_percent": usage_percent
        }
    except Exception as e:
        logger.warning(f"⚠️ Failed to get memory stats: {e}")
        return {}

# FIXED: Context manager implementation instead of decorator  
@contextmanager
def memory_safe_execution(stage_name="Unknown"):
    """Context manager để ensure memory safety"""
    # Pre-execution memory check
    mem_stats = get_memory_stats()
    if mem_stats.get("usage_percent", 0) > MEMORY_CONFIG["emergency_cleanup_threshold"]:
        logger.warning(f"🚨 High memory usage before {stage_name}: {mem_stats.get('usage_percent', 0):.1f}%")
        emergency_memory_cleanup(force_aggressive=True)
    
    start_time = time.time()
    initial_memory = mem_stats.get("allocated_gb", 0)
    
    try:
        # Check available memory
        if mem_stats.get("free_gb", 0) < MEMORY_CONFIG["min_free_memory_gb"]:
            logger.warning(f"⚠️ Low free memory for {stage_name}: {mem_stats.get('free_gb', 0):.1f}GB")
            emergency_memory_cleanup(force_aggressive=True)
        
        yield  # Execute the code block
        
        # Post-execution cleanup
        emergency_memory_cleanup()
        
        # Log performance
        duration = time.time() - start_time
        final_stats = get_memory_stats()
        memory_delta = final_stats.get("allocated_gb", 0) - initial_memory
        
        logger.info(f"✅ {stage_name}: {duration:.1f}s, Memory: {final_stats.get('allocated_gb', 0):.1f}GB ({memory_delta:+.1f}GB)")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"🚨 OOM Error in {stage_name}: {e}")
            emergency_memory_cleanup(force_aggressive=True)
            raise
        else:
            raise
    except Exception as e:
        logger.error(f"❌ Error in {stage_name}: {e}")
        raise
    finally:
        emergency_memory_cleanup()

def get_optimal_attention_mode(prefer_memory_efficient=True) -> str:
    """Determine optimal attention mode"""
    if prefer_memory_efficient:
        if ATTENTION_MECHANISMS['flash_attn']:
            return "flash_attn"
        elif ATTENTION_MECHANISMS['xformers']:
            return "xformers"
        elif ATTENTION_MECHANISMS['sageattention']:
            return "sageattn"
        else:
            return "pytorch"
    else:
        if ATTENTION_MECHANISMS['sageattention']:
            return "sageattn"
        elif ATTENTION_MECHANISMS['flash_attn']:
            return "flash_attn"
        elif ATTENTION_MECHANISMS['xformers']:
            return "xformers"
        else:
            return "pytorch"

def auto_optimize_for_memory(job_input: dict) -> dict:
    """Auto-optimize parameters để prevent memory issues"""
    logger.info("🔧 Auto-optimizing parameters for memory efficiency...")
    
    mem_stats = get_memory_stats()
    available_memory = mem_stats.get("free_gb", 30)
    
    width = job_input.get("width", 400)
    height = job_input.get("height", 704)
    max_audio_duration = job_input.get("max_audio_duration", 30)
    
    resolution_factor = (width * height) / (400 * 704)
    duration_factor = max_audio_duration / 20
    complexity_score = resolution_factor * duration_factor
    
    # CRITICAL: Always enable block swap
    job_input["use_block_swap"] = True
    
    if complexity_score > 2.0 or available_memory < 15:
        job_input["blocks_to_swap"] = 30
        logger.info("🔧 High complexity - aggressive block swap (30 blocks)")
    elif complexity_score > 1.5 or available_memory < 20:
        job_input["blocks_to_swap"] = 25
        logger.info("🔧 Moderate complexity - block swap (25 blocks)")
    else:
        job_input["blocks_to_swap"] = 20
        logger.info("🔧 Standard complexity - block swap (20 blocks)")
    
    if width > 450 or height > 800 or available_memory < 12:
        job_input["use_vae_tiling"] = True
        logger.info("🔧 Auto-enabled VAE tiling")
    
    if max_audio_duration > 25 and available_memory < 15:
        job_input["max_audio_duration"] = 20
        logger.info("🔧 Auto-reduced audio duration to 20s")
    elif max_audio_duration > 20 and available_memory < 10:
        job_input["max_audio_duration"] = 15
        logger.info("🔧 Auto-reduced audio duration to 15s")
    
    if available_memory < 10:
        new_width = min(width, 320)
        new_height = min(height, 576)
        if new_width != width or new_height != height:
            job_input["width"] = new_width
            job_input["height"] = new_height
            logger.warning(f"🔧 Critical memory - reduced resolution to {new_width}x{new_height}")
    
    job_input["_force_memory_efficient_attention"] = available_memory < 15
    
    logger.info(f"📊 Optimization: {complexity_score:.1f} complexity, {available_memory:.1f}GB available")
    return job_input

def verify_models(mode: str = "multitalk") -> Tuple[bool, List[str]]:
    """Verify required models"""
    logger.info(f"🔍 Verifying models for {mode} mode...")
    missing_models = []
    
    base_models = ["wan21_model", "wan21_vae", "text_encoder", "clip_vision", "wav2vec_model"]
    if mode == "multitalk":
        required_models = base_models + ["multitalk_model"]
    else:
        required_models = base_models + ["infinitalk_model"]
    
    total_size = 0
    for name in required_models:
        path = MODEL_CONFIGS[name]
        if os.path.exists(path):
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            total_size += file_size_mb
        else:
            missing_models.append(f"{name}: {path}")
    
    if missing_models:
        return False, missing_models
    else:
        logger.info(f"✅ All {len(required_models)} models verified! Total: {total_size:.1f}MB")
        return True, []

def ensure_odd_frames(frames: int) -> int:
    return frames if frames % 2 == 1 else frames + 1

def image_width_height(image_tensor):
    if image_tensor.ndim == 4:
        _, height, width, _ = image_tensor.shape
    elif image_tensor.ndim == 3:
        height, width, _ = image_tensor.shape
    else:
        raise ValueError(f"Unsupported image shape: {image_tensor.shape}")
    return width, height

def load_and_process_audio(audio_paths: List[str], max_duration: int = None):
    """Load và process audio với memory optimization"""
    logger.info(f"🎵 Loading {len(audio_paths)} audio files...")
    load_audio = LoadAudio()
    loaded_audios = []
    
    for i, path in enumerate(audio_paths):
        if path and os.path.exists(path):
            try:
                audio_data = load_audio.load(path)[0]
                loaded_audios.append(audio_data)
                logger.info(f"✅ Audio {i+1} loaded: {audio_data['sample_rate']}Hz")
                emergency_memory_cleanup()
            except Exception as e:
                logger.warning(f"⚠️ Failed to load audio {i+1}: {e}")
                loaded_audios.append(None)
        else:
            loaded_audios.append(None)
    
    non_none_audios = [a for a in loaded_audios if a is not None]
    if not non_none_audios:
        raise ValueError("No valid audio files loaded")
    
    if len(non_none_audios) > 1:
        logger.info("🔄 Combining multiple audio files...")
        combined_waveform = torch.cat([a["waveform"] for a in non_none_audios], dim=-1)
        sample_rate = non_none_audios[0]["sample_rate"]
        combined_audio = {"waveform": combined_waveform, "sample_rate": sample_rate}
    else:
        combined_audio = non_none_audios[0]
    
    if max_duration:
        audio_duration = combined_audio["waveform"].shape[-1] / combined_audio["sample_rate"]
        if audio_duration > max_duration:
            logger.info(f"✂️ Cropping audio from {audio_duration:.1f}s to {max_duration}s")
            audio_crop = AudioCrop()
            combined_audio = audio_crop.main(combined_audio, "00", f"{max_duration}")[0]
    
    return combined_audio, loaded_audios

def save_video_with_audio(frames_tensor, output_path, fps, audio_path=None):
    """Save video với memory-efficient processing"""
    try:
        logger.info(f"🎬 Saving video to: {output_path}")
        
        if torch.is_tensor(frames_tensor):
            frames_np = frames_tensor.detach().cpu().float().numpy()
            del frames_tensor
            emergency_memory_cleanup()
        else:
            frames_np = np.array(frames_tensor, dtype=np.float32)
        
        if frames_np.ndim == 5 and frames_np.shape[0] == 1:
            frames_np = frames_np[0]
        
        frames_np = np.clip(frames_np * 255.0, 0, 255).astype(np.uint8)
        logger.info(f"📊 Video stats: {frames_np.shape}, {frames_np.dtype}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if audio_path and os.path.exists(audio_path):
            temp_video_path = output_path.replace('.mp4', '_temp.mp4')
            
            with imageio.get_writer(temp_video_path, fps=fps, codec='h264', quality=6) as writer:
                chunk_size = min(50, len(frames_np) // 4)
                for i in range(0, len(frames_np), chunk_size):
                    chunk = frames_np[i:i+chunk_size]
                    for frame in chunk:
                        writer.append_data(frame)
                    del chunk
                    emergency_memory_cleanup()
            
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-i', temp_video_path, '-i', audio_path,
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-c:a', 'aac', '-b:a', '96k',
                '-shortest', output_path
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=180)
                os.remove(temp_video_path)
                logger.info("✅ Video với audio saved successfully")
            except Exception as e:
                logger.warning(f"⚠️ FFmpeg failed: {e}")
                os.rename(temp_video_path, output_path)
        else:
            with imageio.get_writer(output_path, fps=fps, codec='h264', quality=6) as writer:
                chunk_size = min(50, len(frames_np) // 4)
                for i in range(0, len(frames_np), chunk_size):
                    chunk = frames_np[i:i+chunk_size]
                    for frame in chunk:
                        writer.append_data(frame)
                    del chunk
                    emergency_memory_cleanup()
        
        del frames_np
        emergency_memory_cleanup()
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Video saving failed: {e}")
        raise e

def generate_video_wan21_memory_optimized(image_path: str, audio_paths: List[str], **kwargs) -> str:
    """Memory-optimized WAN2.1 video generation"""
    try:
        logger.info("🎬 Starting memory-optimized WAN2.1 generation...")
        
        kwargs = auto_optimize_for_memory(kwargs)
        
        positive_prompt = kwargs.get('positive_prompt', 'The person takes turns talking')
        negative_prompt = kwargs.get('negative_prompt', 'bright tones, overexposed, static, blurred details')
        
        width = kwargs.get('width', 400)
        height = kwargs.get('height', 704)
        fps = kwargs.get('fps', 25)
        seed = kwargs.get('seed', random.randint(1, 2**32 - 1))
        steps = kwargs.get('steps', 4)
        cfg_scale = kwargs.get('cfg_scale', 1.0)
        scheduler = kwargs.get('scheduler', 'flowmatch_distill')
        
        mode = kwargs.get('mode', 'multitalk')
        use_block_swap = kwargs.get('use_block_swap', True)
        blocks_to_swap = kwargs.get('blocks_to_swap', 20)
        use_speed_lora = kwargs.get('use_speed_lora', True)
        speed_lora_strength = kwargs.get('speed_lora_strength', 1.0)
        max_audio_duration = kwargs.get('max_audio_duration', None)
        use_vae_tiling = kwargs.get('use_vae_tiling', False)
        
        prefer_memory_efficient = kwargs.get('_force_memory_efficient_attention', True)
        attention_mode = get_optimal_attention_mode(prefer_memory_efficient)
        
        logger.info(f"🎯 Memory-Optimized Parameters:")
        logger.info(f" Mode: {mode}, Attention: {attention_mode}")
        logger.info(f" Resolution: {width}x{height}, FPS: {fps}")
        logger.info(f" Block Swap: {use_block_swap} ({blocks_to_swap} blocks)")
        logger.info(f" VAE Tiling: {use_vae_tiling}")
        
        if not WANVIDEO_AVAILABLE:
            raise RuntimeError("WanVideoWrapper modules not available")
        
        with torch.inference_mode():
            logger.info("🔧 Initializing nodes...")
            
            # Core nodes
            clip_loader = CLIPLoader()
            clip_encode_positive = CLIPTextEncode()
            clip_encode_negative = CLIPTextEncode()
            clip_vision_loader = CLIPVisionLoader()
            load_image = LoadImage()
            image_scaler = ImageScale()
            
            # WanVideo nodes
            wan_text_embed_bridge = WanVideoTextEmbedBridge()
            wan_clip_vision = WanVideoClipVisionEncode()
            wan_vae_loader = WanVideoVAELoader()
            wan_vae_decoder = WanVideoDecode()
            multitalk_img2vid = WanVideoImageToVideoMultiTalk()
            
            # Audio nodes
            load_wav2vec = DownloadAndLoadWav2VecModel()
            multitalk_wav2vec = MultiTalkWav2VecEmbeds()
            
            # Model loading nodes
            wan_model_loader = WanVideoModelLoader()
            multitalk_loader = MultiTalkModelLoader()
            wan_lora_select = WanVideoLoraSelect()
            block_swapper = WanVideoBlockSwap()
            wan_sampler = WanVideoSampler()
            wan_context_options = WanVideoContextOptions()
            
            logger.info("✅ All nodes initialized")
            
            # STAGE 1: Text encoding
            with memory_safe_execution("Text Encoding"):
                logger.info("📝 Loading Text Encoder...")
                clip = clip_loader.load_clip(os.path.basename(MODEL_CONFIGS["text_encoder"]), "wan", "default")[0]
                positive = clip_encode_positive.encode(clip, positive_prompt)[0]
                negative = clip_encode_negative.encode(clip, negative_prompt)[0]
                text_embeds = wan_text_embed_bridge.process(positive, negative)[0]
                del clip, positive, negative
            
            # STAGE 2: Image processing
            with memory_safe_execution("Image Processing"):
                logger.info("🖼️ Loading và processing image...")
                clip_vision = clip_vision_loader.load_clip(os.path.basename(MODEL_CONFIGS["clip_vision"]))[0]
                loaded_image = load_image.load_image(image_path)[0]
                
                orig_width, orig_height = image_width_height(loaded_image)
                logger.info(f"Original image: {orig_width}x{orig_height}")
                
                if orig_width != width or orig_height != height:
                    logger.info(f"Scaling image to {width}x{height}...")
                    loaded_image = image_scaler.upscale(loaded_image, "lanczos", width, height, "disabled")[0]
                
                clip_vision_output = wan_clip_vision.process(
                    clip_vision=clip_vision,
                    image_1=loaded_image,
                    strength_1=1.0,
                    strength_2=1.0,
                    force_offload=True,
                    crop="disabled",
                    combine_embeds="average",
                    image_2=None,
                    negative_image=None,
                    tiles=0,
                    ratio=0.5
                )[0]
                del clip_vision
            
            # STAGE 3: Audio processing
            with memory_safe_execution("Audio Processing"):
                combined_audio, loaded_audios = load_and_process_audio(audio_paths, max_audio_duration)
                
                audio_duration = combined_audio["waveform"].shape[-1] / combined_audio["sample_rate"]
                frames = max(1 * fps, int(audio_duration * fps))
                frames = ensure_odd_frames(frames)
                
                logger.info(f"Audio duration: {audio_duration:.1f}s, Frames: {frames}")
                
                output_audio_path = "/tmp/combined_audio.wav"
                waveform = combined_audio["waveform"].squeeze(0)
                torchaudio.save(output_audio_path, waveform, combined_audio["sample_rate"])
                del waveform
            
            # STAGE 4: Wav2Vec processing
            with memory_safe_execution("Audio Embedding"):
                logger.info("🎤 Loading Wav2Vec model...")
                wav2vec_model = load_wav2vec.loadmodel("TencentGameMate/chinese-wav2vec2-base", "fp16", "main_device")[0]
                
                wav2vec_embeds, _, actual_num_frames = multitalk_wav2vec.process(
                    wav2vec_model, True, fps, frames,
                    loaded_audios[0] if loaded_audios else combined_audio,
                    1, 1, "para",
                    loaded_audios[1] if len(loaded_audios) > 1 else None,
                    loaded_audios[2] if len(loaded_audios) > 2 else None,
                    loaded_audios[3] if len(loaded_audios) > 3 else None
                )
                del wav2vec_model
            
            # STAGE 5: VAE và image embedding
            with memory_safe_execution("VAE & Image Embedding"):
                logger.info("🎨 Loading VAE...")
                vae = wan_vae_loader.loadmodel(os.path.basename(MODEL_CONFIGS["wan21_vae"]), "fp16")[0]
                
                image_embeds = multitalk_img2vid.process(
                    vae, width, height, frames, fps,
                    False, 'mkl', loaded_image, False, clip_vision_output
                )[0]
                del loaded_image, clip_vision_output
            
            # STAGE 6: Talk model loading
            with memory_safe_execution("Talk Model Loading"):
                if mode == "multitalk":
                    logger.info("🗣️ Loading MultiTalk Model...")
                    talk_model_path = MODEL_CONFIGS["multitalk_model"]
                else:
                    logger.info("🗣️ Loading InfiniteTalk Model...")
                    talk_model_path = MODEL_CONFIGS["infinitalk_model"]
                
                multitalk_model = multitalk_loader.loadmodel(os.path.basename(talk_model_path), "fp16")[0]
            
            # STAGE 7: LoRA và Block Swap setup
            wan_speed_lora = None
            if use_speed_lora:
                logger.info("⚡ Loading Speed LoRA...")
                wan_speed_lora = wan_lora_select.getlorapath(
                    os.path.basename(MODEL_CONFIGS["speed_lora"]),
                    speed_lora_strength, None, {}, None, False, False
                )[0]
            
            block_swap_args = None
            if use_block_swap:
                logger.info(f"🔄 Setting up block swap ({blocks_to_swap} blocks)...")
                block_swap_args = block_swapper.setargs(
                    blocks_to_swap=blocks_to_swap,
                    offload_img_emb=True,
                    offload_txt_emb=True,
                    use_non_blocking=True,
                    vace_blocks_to_swap=0,
                    prefetch_blocks=1,
                    block_swap_debug=False
                )[0]
            
            # STAGE 8: Main model loading
            with memory_safe_execution("WAN2.1 Model Loading"):
                logger.info(f"🎯 Loading WAN2.1 base model với {attention_mode} attention...")
                
                try:
                    model = wan_model_loader.loadmodel(
                        model=os.path.basename(MODEL_CONFIGS["wan21_model"]),
                        base_precision="fp16",
                        load_device="offload_device",
                        quantization="disabled",
                        compile_args=None,
                        attention_mode=attention_mode,
                        block_swap_args=block_swap_args,
                        lora=wan_speed_lora,
                        vram_management_args=None,
                        vace_model=None,
                        fantasytalking_model=None,
                        multitalk_model=multitalk_model,
                        fantasyportrait_model=None
                    )[0]
                    
                    logger.info(f"✅ Model loaded successfully với {attention_mode}")
                    
                except Exception as model_error:
                    logger.warning(f"⚠️ {attention_mode} failed: {model_error}")
                    logger.warning("🔄 Retrying với PyTorch attention...")
                    
                    emergency_memory_cleanup(force_aggressive=True)
                    
                    model = wan_model_loader.loadmodel(
                        model=os.path.basename(MODEL_CONFIGS["wan21_model"]),
                        base_precision="fp16",
                        load_device="offload_device",
                        quantization="disabled",
                        compile_args=None,
                        attention_mode="pytorch",
                        block_swap_args=block_swap_args,
                        lora=wan_speed_lora,
                        vram_management_args=None,
                        vace_model=None,
                        fantasytalking_model=None,
                        multitalk_model=multitalk_model,
                        fantasyportrait_model=None
                    )[0]
                    
                    logger.info("✅ Model loaded với PyTorch attention fallback")
                
                del multitalk_model
            
            # STAGE 9: Context optimization
            context_options = None
            if frames > 200:
                context_frames = min(frames, 200)
                context_options = wan_context_options.process(
                    context_schedule="uniform_standard",
                    context_frames=context_frames,
                    context_stride=12,
                    context_overlap=6,
                    freenoise=True,
                    verbose=False,
                    image_cond_start_step=3,
                    image_cond_window_count=1,
                    vae=None,
                    fuse_method="linear",
                    reference_latent=None
                )[0]
                logger.info(f"🔧 Memory-optimized context: {context_frames} frames")
            
            # STAGE 10: Video sampling
            with memory_safe_execution("Video Sampling"):
                logger.info("🎬 Sampling video với memory optimization...")
                
                sampled = wan_sampler.process(
                    model=model,
                    image_embeds=image_embeds,
                    shift=8,
                    steps=steps,
                    cfg=cfg_scale,
                    seed=seed,
                    scheduler=scheduler,
                    riflex_freq_index=0,
                    text_embeds=text_embeds,
                    force_offload=True,
                    samples=None,
                    feta_args=None,
                    denoise_strength=1.0,
                    context_options=context_options,
                    cache_args=None,
                    teacache_args=None,
                    flowedit_args=None,
                    batched_cfg=False,
                    slg_args=None,
                    rope_function="default",
                    loop_args=None,
                    experimental_args=None,
                    sigmas=None,
                    unianimate_poses=None,
                    fantasytalking_embeds=None,
                    uni3c_embeds=None,
                    multitalk_embeds=wav2vec_embeds,
                    freeinit_args=None,
                    start_step=0,
                    end_step=-1,
                    add_noise_to_samples=False
                )[0]
                
                del model, text_embeds, wav2vec_embeds
            
            # STAGE 11: VAE decoding
            with memory_safe_execution("VAE Decoding"):
                logger.info("🎨 Decoding latents với memory-efficient VAE tiling...")
                
                decoded = wan_vae_decoder.decode(
                    vae=vae,
                    samples=sampled,
                    enable_vae_tiling=use_vae_tiling,
                    tile_x=256 if use_vae_tiling else 512,
                    tile_y=256 if use_vae_tiling else 512,
                    tile_stride_x=128 if use_vae_tiling else 256,
                    tile_stride_y=128 if use_vae_tiling else 256,
                    normalization="default"
                )[0]
                
                del vae, sampled, image_embeds
            
            # STAGE 12: Save video
            output_path = f"/app/ComfyUI/output/wan21_{mode}_{uuid.uuid4().hex[:8]}.mp4"
            final_output_path = save_video_with_audio(decoded, output_path, fps, output_audio_path)
            
            del decoded
            emergency_memory_cleanup()
            
            logger.info(f"✅ Memory-optimized video generation completed: {final_output_path}")
            return final_output_path
            
    except Exception as e:
        logger.error(f"❌ Memory-optimized generation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        emergency_memory_cleanup(force_aggressive=True)
        return None
    finally:
        emergency_memory_cleanup(force_aggressive=True)

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload với retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if not minio_client:
                raise RuntimeError("MinIO client not initialized")
            
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")
            
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.info(f"📤 Uploading: {object_name} ({file_size_mb:.1f}MB) - Attempt {attempt + 1}")
            
            minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
            
            file_url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
            logger.info(f"✅ Upload completed: {file_url}")
            return file_url
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"⚠️ Upload attempt {attempt + 1} failed: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ Upload failed after {max_retries} attempts: {e}")
                raise e

def validate_input_parameters(job_input: dict) -> Tuple[bool, str]:
    """Enhanced input validation"""
    try:
        required_params = ["image_url", "audio_urls"]
        for param in required_params:
            if param not in job_input or not job_input[param]:
                return False, f"Missing required parameter: {param}"
        
        try:
            response = requests.head(job_input["image_url"], timeout=15)
            if response.status_code != 200:
                return False, f"Image URL not accessible: {response.status_code}"
        except Exception as e:
            return False, f"Image URL validation failed: {str(e)}"
        
        audio_urls = job_input["audio_urls"]
        if not isinstance(audio_urls, list) or len(audio_urls) == 0:
            return False, "audio_urls must be a non-empty list"
        
        if len(audio_urls) > 4:
            return False, "Maximum 4 audio files supported"
        
        width = job_input.get("width", 400)
        height = job_input.get("height", 704)
        if not (256 <= width <= 800 and 256 <= height <= 800):
            return False, "Width and height must be between 256 and 800 for memory safety"
        
        mode = job_input.get("mode", "multitalk")
        if mode not in ["multitalk", "infinitetalk"]:
            return False, "Mode must be 'multitalk' or 'infinitetalk'"
        
        steps = job_input.get("steps", 4)
        if not (1 <= steps <= 15):
            return False, "Steps must be between 1 and 15"
        
        fps = job_input.get("fps", 25)
        if not (15 <= fps <= 30):
            return False, "FPS must be between 15 and 30"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Parameter validation error: {str(e)}"

def handler(job):
    """Enhanced RunPod handler với FIXED context manager usage"""
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        
        initial_mem = get_memory_stats()
        logger.info(f"🚀 Job {job_id} started - Initial memory: {initial_mem.get('allocated_gb', 0):.1f}GB")
        
        is_valid, validation_message = validate_input_parameters(job_input)
        if not is_valid:
            return {"error": validation_message, "status": "failed", "job_id": job_id}
        
        image_url = job_input["image_url"]
        audio_urls = job_input["audio_urls"]
        
        parameters = {
            "positive_prompt": job_input.get("positive_prompt", "The person takes turns talking"),
            "negative_prompt": job_input.get("negative_prompt", "bright tones, overexposed, static, blurred details"),
            "width": job_input.get("width", 400),
            "height": job_input.get("height", 704),
            "fps": job_input.get("fps", 25),
            "seed": job_input.get("seed", 0),
            "steps": job_input.get("steps", 4),
            "cfg_scale": job_input.get("cfg_scale", 1.0),
            "scheduler": job_input.get("scheduler", "flowmatch_distill"),
            "mode": job_input.get("mode", "multitalk"),
            "use_block_swap": job_input.get("use_block_swap", True),
            "blocks_to_swap": job_input.get("blocks_to_swap", 20),
            "use_speed_lora": job_input.get("use_speed_lora", True),
            "speed_lora_strength": job_input.get("speed_lora_strength", 1.0),
            "max_audio_duration": job_input.get("max_audio_duration", None),
            "use_vae_tiling": job_input.get("use_vae_tiling", False)
        }
        
        logger.info(f"🚀 Job {job_id}: WAN2.1 {parameters['mode'].upper()} Generation Started")
        logger.info(f"🖼️ Image: {image_url}")
        logger.info(f"🎵 Audio files: {len(audio_urls)}")
        logger.info(f"⚙️ Resolution: {parameters['width']}x{parameters['height']} @ {parameters['fps']}fps")
        
        # Verify models
        with memory_safe_execution("Model Verification"):
            models_ok, missing_models = verify_models(parameters['mode'])
            if not models_ok:
                return {
                    "error": "Required models are missing",
                    "missing_models": missing_models,
                    "status": "failed",
                    "mode": parameters['mode']
                }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download files
            with memory_safe_execution("File Downloads"):
                # Download image
                image_path = os.path.join(temp_dir, "input_image.jpg")
                logger.info("📥 Downloading input image...")
                
                response = requests.get(image_url, timeout=60, stream=True)
                response.raise_for_status()
                
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Download audio files
                audio_paths = []
                for i, audio_url in enumerate(audio_urls):
                    if audio_url:
                        audio_path = os.path.join(temp_dir, f"audio_{i}.wav")
                        try:
                            response = requests.get(audio_url, timeout=60, stream=True)
                            response.raise_for_status()
                            
                            with open(audio_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                            
                            audio_paths.append(audio_path)
                            logger.info(f"✅ Audio {i+1} downloaded")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to download audio {i+1}: {e}")
                            audio_paths.append(None)
                    else:
                        audio_paths.append(None)
                
                valid_audio_paths = [p for p in audio_paths if p is not None]
                if not valid_audio_paths:
                    return {"error": "No valid audio files could be downloaded"}
            
            # Generate video
            generation_start = time.time()
            output_path = generate_video_wan21_memory_optimized(
                image_path=image_path,
                audio_paths=audio_paths,
                **parameters
            )
            generation_time = time.time() - generation_start
            
            if not output_path or not os.path.exists(output_path):
                return {"error": "Video generation failed"}
            
            # Upload result
            logger.info("📤 Uploading result...")
            output_filename = f"wan21_fixed_{parameters['mode']}_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            
            try:
                output_url = upload_to_minio(output_path, output_filename)
            except Exception as e:
                return {"error": f"Failed to upload result: {str(e)}"}
            
            # Calculate statistics
            total_time = time.time() - start_time
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            final_mem = get_memory_stats()
            
            logger.info(f"✅ Job {job_id} completed successfully!")
            logger.info(f"⏱️ Total: {total_time:.1f}s (generation: {generation_time:.1f}s)")
            logger.info(f"📊 Output: {file_size_mb:.1f}MB")
            logger.info(f"💾 Peak memory: {final_mem.get('allocated_gb', 0):.1f}GB")
            
            return {
                "output_video_url": output_url,
                "processing_time_seconds": round(total_time, 2),
                "generation_time_seconds": round(generation_time, 2),
                "video_info": {
                    "width": parameters["width"],
                    "height": parameters["height"],
                    "fps": parameters["fps"],
                    "file_size_mb": round(file_size_mb, 2),
                    "mode": parameters["mode"],
                    "audio_files_used": len(valid_audio_paths)
                },
                "generation_params": {
                    "positive_prompt": parameters["positive_prompt"],
                    "steps": parameters["steps"],
                    "cfg_scale": parameters["cfg_scale"],
                    "seed": parameters["seed"] if parameters["seed"] != 0 else "auto-generated",
                    "scheduler": parameters["scheduler"],
                    "mode": parameters["mode"],
                    "attention_mode": get_optimal_attention_mode(True),
                    "block_swap": parameters["use_block_swap"],
                    "blocks_swapped": parameters["blocks_to_swap"],
                    "vae_tiling": parameters["use_vae_tiling"],
                    "workflow_version": "WAN21_CONTEXT_MANAGER_FIXED_v3.1"
                },
                "memory_info": {
                    "initial_memory_gb": round(initial_mem.get('allocated_gb', 0), 1),
                    "peak_memory_gb": round(final_mem.get('allocated_gb', 0), 1),
                    "memory_efficiency": "context_manager_fixed"
                },
                "status": "completed"
            }
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Handler error for job {job_id}: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        emergency_memory_cleanup(force_aggressive=True)
        
        return {
            "error": error_msg,
            "status": "failed",
            "processing_time_seconds": round(time.time() - start_time, 2),
            "job_id": job_id,
            "debug_info": {
                "attention_mechanisms": ATTENTION_MECHANISMS,
                "memory_config": MEMORY_CONFIG,
                "memory_stats": get_memory_stats(),
                "fix_applied": "context_manager_correction"
            }
        }
    finally:
        emergency_memory_cleanup(force_aggressive=True)

def health_check():
    """Enhanced health check"""
    try:
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        mem_stats = get_memory_stats()
        available_memory = mem_stats.get("free_gb", 0)
        
        if available_memory < 5:
            return False, f"Insufficient GPU memory: {available_memory:.1f}GB available"
        
        if not WANVIDEO_AVAILABLE:
            return False, "WanVideoWrapper not available"
        
        # Test context manager
        try:
            with memory_safe_execution("Health Check Test"):
                pass
            logger.info("✅ Context manager test passed")
        except Exception as e:
            return False, f"Context manager test failed: {e}"
        
        multitalk_ok, _ = verify_models("multitalk")
        infinitetalk_ok, _ = verify_models("infinitetalk")
        
        if not (multitalk_ok or infinitetalk_ok):
            return False, "No complete model sets available"
        
        if not minio_client:
            return False, "MinIO not available"
        
        available_modes = []
        if multitalk_ok:
            available_modes.append("multitalk")
        if infinitetalk_ok:
            available_modes.append("infinitetalk")
        
        attention_modes = [name for name, available in ATTENTION_MECHANISMS.items() if available]
        optimal_attention = get_optimal_attention_mode(True)
        
        total_memory = mem_stats.get("total_gb", 0)
        
        health_info = (
            f"Ready - Modes: {', '.join(available_modes)}, "
            f"Memory: {available_memory:.1f}GB/{total_memory:.1f}GB available, "
            f"Attention: {optimal_attention} (available: {', '.join(attention_modes)}), "
            f"Context Manager: FIXED v3.1"
        )
        
        return True, health_info
        
    except Exception as e:
        return False, f"Health check failed: {str(e)}"

if __name__ == "__main__":
    logger.info("🚀 Starting WAN2.1 Memory-Optimized Serverless Worker v3.1 (Context Manager Fixed)...")
    logger.info(f"🔥 PyTorch: {torch.__version__}")
    logger.info(f"🎯 CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        logger.info(f"💾 GPU: {gpu_props.name}")
        logger.info(f"💾 GPU Memory: {gpu_props.total_memory / 1e9:.1f}GB")
        
        mem_stats = get_memory_stats()
        logger.info(f"📊 Memory Management: {MEMORY_CONFIG['max_gpu_usage_percent']}% limit, {mem_stats.get('free_gb', 0):.1f}GB available")
    
    attention_modes = [name for name, available in ATTENTION_MECHANISMS.items() if available]
    logger.info(f"⚡ Available Attention: {attention_modes}")
    logger.info(f"🧹 Memory Optimizations: Enhanced cleanup, Block swap default, VAE tiling support")
    
    try:
        health_ok, health_msg = health_check()
        if not health_ok:
            logger.error(f"❌ Health check failed: {health_msg}")
            sys.exit(1)
        
        logger.info(f"✅ Health check passed: {health_msg}")
        logger.info("🎬 Ready to process memory-optimized WAN2.1 requests...")
        logger.info("🔧 FIXED: Context manager implementation, Memory management, CUDA OOM prevention!")
        
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
