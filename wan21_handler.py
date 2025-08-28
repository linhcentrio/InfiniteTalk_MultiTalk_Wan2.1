#!/usr/bin/env python3
"""
RunPod Serverless Handler cho WAN2.1 + InfiniteTalk/MultiTalk
Dựa trên workflow từ wan21_based_InfiniteTalk_&_MultiTalk.ipynb
Hỗ trợ audio-driven video generation với multiple speakers
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add ComfyUI paths
sys.path.insert(0, '/app/ComfyUI')

# Import ComfyUI components with comprehensive error handling
try:
    # Core ComfyUI nodes
    from nodes import (
        CLIPLoader, CLIPTextEncode, LoadImage, CLIPVisionLoader, ImageScale
    )
    
    # WanVideoWrapper nodes (chính cho workflow này)
    from custom_nodes.ComfyUI_WanVideoWrapper.nodes_model_loading import (
        WanVideoModelLoader, WanVideoVAELoader, WanVideoLoraSelect, WanVideoBlockSwap
    )
    from custom_nodes.ComfyUI_WanVideoWrapper.nodes import (
        WanVideoSampler, WanVideoTextEmbedBridge, WanVideoDecode, WanVideoClipVisionEncode
    )
    from custom_nodes.ComfyUI_WanVideoWrapper.multitalk.nodes import (
        MultiTalkModelLoader, MultiTalkWav2VecEmbeds, WanVideoImageToVideoMultiTalk
    )
    from custom_nodes.ComfyUI_WanVideoWrapper.fantasytalking.nodes import DownloadAndLoadWav2VecModel
    
    # Audio processing nodes
    from comfy_extras.nodes_audio import LoadAudio
    from custom_nodes.audio_separation_nodes_comfyui.src.separation import AudioSeparation
    from custom_nodes.audio_separation_nodes_comfyui.src.crop import AudioCrop
    
    logger.info("✅ All WanVideoWrapper modules imported successfully")
    WANVIDEO_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ WanVideoWrapper import error: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    WANVIDEO_AVAILABLE = False

# MinIO Configuration
MINIO_ENDPOINT = "media.aiclip.ai"
MINIO_ACCESS_KEY = "VtZ6MUPfyTOH3qSiohA2"
MINIO_SECRET_KEY = "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t"
MINIO_BUCKET = "video"
MINIO_SECURE = False

# Initialize MinIO client with error handling
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )
    logger.info("✅ MinIO client initialized")
except Exception as e:
    logger.error(f"❌ MinIO initialization failed: {e}")
    minio_client = None

# Model configurations từ ENV variables
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

# Enable PyTorch optimizations (theo notebook)
try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    # Enable Flash Attention
    torch.backends.cuda.flash_sdp_enabled = True
    torch.backends.cuda.mem_efficient_sdp_enabled = True
    logger.info("✅ PyTorch optimizations enabled")
except Exception as e:
    logger.warning(f"⚠️ PyTorch optimizations partially failed: {e}")

def verify_models(mode: str = "multitalk") -> tuple[bool, list]:
    """Verify required models exist for specified mode"""
    logger.info(f"🔍 Verifying models for {mode} mode...")
    missing_models = []
    existing_models = []
    
    # Base models required for all modes
    base_models = ["wan21_model", "wan21_vae", "text_encoder", "clip_vision", "wav2vec_model"]
    
    # Add talk model based on mode
    if mode == "multitalk":
        required_models = base_models + ["multitalk_model"]
    else:  # infinitetalk
        required_models = base_models + ["infinitalk_model"]
    
    total_size = 0
    for name in required_models:
        path = MODEL_CONFIGS[name]
        if os.path.exists(path):
            try:
                file_size_mb = os.path.getsize(path) / (1024 * 1024)
                existing_models.append(f"{name}: {file_size_mb:.1f}MB")
                total_size += file_size_mb
                logger.info(f"✅ {name}: {file_size_mb:.1f}MB")
            except Exception as e:
                logger.error(f"❌ Error checking {name}: {e}")
                missing_models.append(f"{name}: {path} (error reading)")
        else:
            missing_models.append(f"{name}: {path}")
            logger.error(f"❌ Missing: {name} at {path}")
    
    if missing_models:
        logger.error(f"❌ Missing {len(missing_models)} models for {mode}")
        return False, missing_models
    else:
        logger.info(f"✅ All {len(existing_models)} models verified! Total: {total_size:.1f}MB")
        return True, []

def clear_memory():
    """Enhanced memory cleanup theo notebook"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            torch.cuda.synchronize()
        except:
            pass

def ensure_odd_frames(frames: int) -> int:
    """Ensure frame count is odd như trong notebook"""
    return frames if frames % 2 == 1 else frames + 1

def image_width_height(image_tensor):
    """Get width và height từ image tensor theo notebook"""
    if image_tensor.ndim == 4:
        _, height, width, _ = image_tensor.shape
    elif image_tensor.ndim == 3:
        height, width, _ = image_tensor.shape
    else:
        raise ValueError(f"Unsupported image shape: {image_tensor.shape}")
    return width, height

def load_and_process_audio(audio_paths: list, max_duration: int = None):
    """Load và process multiple audio files theo notebook logic"""
    logger.info(f"🎵 Loading {len(audio_paths)} audio files...")
    load_audio = LoadAudio()
    loaded_audios = []
    
    for i, path in enumerate(audio_paths):
        if path and os.path.exists(path):
            try:
                audio_data = load_audio.load(path)[0]
                loaded_audios.append(audio_data)
                logger.info(f"✅ Audio {i+1} loaded: {audio_data['sample_rate']}Hz")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load audio {i+1}: {e}")
                loaded_audios.append(None)
        else:
            loaded_audios.append(None)
    
    # Process multiple audios theo notebook logic
    non_none_audios = [a for a in loaded_audios if a is not None]
    if not non_none_audios:
        raise ValueError("No valid audio files loaded")
    
    if len(non_none_audios) > 1:
        # Combine multiple audios
        logger.info("🔄 Combining multiple audio files...")
        combined_waveform = torch.cat([a["waveform"] for a in non_none_audios], dim=-1)
        sample_rate = non_none_audios[0]["sample_rate"]
        combined_audio = {"waveform": combined_waveform, "sample_rate": sample_rate}
    else:
        combined_audio = non_none_audios[0]
    
    # Crop if max duration specified
    if max_duration:
        audio_duration = combined_audio["waveform"].shape[-1] / combined_audio["sample_rate"]
        if audio_duration > max_duration:
            logger.info(f"✂️ Cropping audio from {audio_duration:.1f}s to {max_duration}s")
            audio_crop = AudioCrop()
            combined_audio = audio_crop.main(combined_audio, "00", f"{max_duration}")[0]
    
    return combined_audio, loaded_audios

def save_video_with_audio(frames_tensor, output_path, fps, audio_path=None):
    """Save video frames với audio merge theo notebook logic"""
    try:
        logger.info(f"🎬 Saving video to: {output_path}")
        
        # Convert tensor to numpy frames
        if torch.is_tensor(frames_tensor):
            frames_np = frames_tensor.detach().cpu().float().numpy()
        else:
            frames_np = np.array(frames_tensor, dtype=np.float32)
        
        # Handle batch dimension
        if frames_np.ndim == 5 and frames_np.shape[0] == 1:
            frames_np = frames_np[0]
        
        # Convert to uint8
        frames_np = (frames_np * 255.0).astype(np.uint8)
        
        logger.info(f"📊 Video stats: {frames_np.shape}, {frames_np.dtype}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if audio_path and os.path.exists(audio_path):
            # Save video with audio using ffmpeg
            temp_video_path = output_path.replace('.mp4', '_temp.mp4')
            
            # Save video without audio first
            with imageio.get_writer(temp_video_path, fps=fps) as writer:
                for frame in frames_np:
                    writer.append_data(frame)
            
            # Merge with audio using ffmpeg
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                output_path
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                os.remove(temp_video_path)
                logger.info("✅ Video với audio saved successfully")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ FFmpeg failed: {e}")
                # Fallback: rename temp file
                os.rename(temp_video_path, output_path)
        else:
            # Save video without audio
            with imageio.get_writer(output_path, fps=fps) as writer:
                for frame in frames_np:
                    writer.append_data(frame)
            logger.info("✅ Video saved without audio")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Video saving failed: {e}")
        raise e

def generate_video_wan21_complete(image_path: str, audio_paths: list, **kwargs) -> str:
    """
    COMPLETE WAN2.1 video generation với InfiniteTalk/MultiTalk
    Theo exact workflow từ notebook
    """
    try:
        logger.info("🎬 Starting WAN2.1 InfiniteTalk/MultiTalk generation...")
        
        # Extract parameters với default values từ notebook
        positive_prompt = kwargs.get('positive_prompt', 'The person takes turns talking')
        negative_prompt = kwargs.get('negative_prompt', 'bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards')
        
        # Video settings
        width = kwargs.get('width', 400)
        height = kwargs.get('height', 704)
        fps = kwargs.get('fps', 25)
        seed = kwargs.get('seed', random.randint(1, 2**32 - 1))
        steps = kwargs.get('steps', 4)
        cfg_scale = kwargs.get('cfg_scale', 1.0)
        scheduler = kwargs.get('scheduler', 'flowmatch_distill')  # Theo notebook default
        
        # Advanced settings
        mode = kwargs.get('mode', 'multitalk')  # multitalk hoặc infinitetalk
        use_block_swap = kwargs.get('use_block_swap', False)
        blocks_to_swap = kwargs.get('blocks_to_swap', 20)
        use_speed_lora = kwargs.get('use_speed_lora', True)
        speed_lora_strength = kwargs.get('speed_lora_strength', 1.0)
        max_audio_duration = kwargs.get('max_audio_duration', None)
        
        logger.info(f"🎯 Generation Parameters:")
        logger.info(f" Mode: {mode}")
        logger.info(f" Resolution: {width}x{height}")
        logger.info(f" FPS: {fps}, Steps: {steps}")
        logger.info(f" CFG Scale: {cfg_scale}, Seed: {seed}")
        logger.info(f" Audio files: {len(audio_paths)}")
        
        # Verify WanVideoWrapper availability
        if not WANVIDEO_AVAILABLE:
            raise RuntimeError("WanVideoWrapper modules not available")
        
        with torch.inference_mode():
            # Initialize tất cả WanVideoWrapper nodes
            logger.info("🔧 Initializing WanVideoWrapper nodes...")
            
            # Core nodes
            clip_loader = CLIPLoader()
            clip_encode_positive = CLIPTextEncode()
            clip_encode_negative = CLIPTextEncode()
            clip_vision_loader = CLIPVisionLoader()
            load_image = LoadImage()
            image_scaler = ImageScale()
            
            # WanVideo specific nodes
            wan_text_embed_bridge = WanVideoTextEmbedBridge()
            wan_clip_vision = WanVideoClipVisionEncode()
            wan_vae_loader = WanVideoVAELoader()
            wan_vae_decoder = WanVideoDecode()
            multitalk_img2vid = WanVideoImageToVideoMultiTalk()
            
            # Audio processing nodes
            load_wav2vec = DownloadAndLoadWav2VecModel()
            multitalk_wav2vec = MultiTalkWav2VecEmbeds()
            
            # Model loading nodes
            wan_model_loader = WanVideoModelLoader()
            multitalk_loader = MultiTalkModelLoader()
            wan_lora_select = WanVideoLoraSelect()
            block_swapper = WanVideoBlockSwap()
            
            # Sampling và decoding
            wan_sampler = WanVideoSampler()
            
            logger.info("✅ All nodes initialized")
            
            # STEP 1: Load text encoder và encode prompts (EXACT theo notebook)
            logger.info("📝 Loading Text Encoder...")
            clip = clip_loader.load_clip(os.path.basename(MODEL_CONFIGS["text_encoder"]), "wan", "default")[0]
            
            positive = clip_encode_positive.encode(clip, positive_prompt)[0]
            negative = clip_encode_negative.encode(clip, negative_prompt)[0]
            
            # Create text embeds bridge
            text_embeds = wan_text_embed_bridge.process(positive, negative)[0]
            
            del clip
            clear_memory()
            
            # STEP 2: Load và process image (EXACT theo notebook)
            logger.info("🖼️ Loading và processing image...")
            clip_vision = clip_vision_loader.load_clip(os.path.basename(MODEL_CONFIGS["clip_vision"]))[0]
            
            loaded_image = load_image.load_image(image_path)[0]
            
            # Get original dimensions
            orig_width, orig_height = image_width_height(loaded_image)
            logger.info(f"Original image: {orig_width}x{orig_height}")
            
            # Scale image to target resolution
            logger.info(f"Scaling image to {width}x{height}...")
            loaded_image = image_scaler.upscale(
                loaded_image, "lanczos", width, height, "disabled"
            )[0]
            
            # Process với CLIP Vision
            logger.info("👁️ Processing với CLIP Vision...")
            clip_vision_output = wan_clip_vision.process(
                clip_vision=clip_vision,
                image_1=loaded_image,
                strength_1=1.0,
                strength_2=1.0,
                force_offload=False,
                crop="disabled",
                combine_embeds="average",
                image_2=None,
                negative_image=None,
                tiles=0,
                ratio=0.5
            )[0]
            
            del clip_vision
            clear_memory()
            
            # STEP 3: Load và process audio files (EXACT theo notebook)
            logger.info("🎵 Loading và processing audio...")
            combined_audio, loaded_audios = load_and_process_audio(audio_paths, max_audio_duration)
            
            # Calculate frames from audio duration
            audio_duration = combined_audio["waveform"].shape[-1] / combined_audio["sample_rate"]
            frames = max(1 * fps, int(audio_duration * fps))
            frames = ensure_odd_frames(frames)  # Ensure odd frames theo notebook
            
            logger.info(f"Audio duration: {audio_duration:.1f}s, Frames: {frames}")
            
            # Save combined audio for later use
            output_audio_path = "/tmp/combined_audio.wav"
            waveform = combined_audio["waveform"].squeeze(0)
            torchaudio.save(output_audio_path, waveform, combined_audio["sample_rate"])
            
            # STEP 4: Load Wav2Vec và create audio embeddings
            logger.info("🎤 Loading Wav2Vec model...")
            wav2vec_model = load_wav2vec.loadmodel("TencentGameMate/chinese-wav2vec2-base", "fp16", "main_device")[0]
            
            logger.info("🔄 Creating audio embeddings...")
            wav2vec_embeds, _, actual_num_frames = multitalk_wav2vec.process(
                wav2vec_model,
                True,  # auto_sync
                fps,
                frames,
                loaded_audios[0] if loaded_audios else combined_audio,
                1,  # strength_1
                1,  # strength_2
                "para",  # mode
                loaded_audios[1] if len(loaded_audios) > 1 else None,
                loaded_audios[2] if len(loaded_audios) > 2 else None,
                loaded_audios[3] if len(loaded_audios) > 3 else None
            )
            
            del wav2vec_model
            clear_memory()
            
            # STEP 5: Load VAE và create image embeds
            logger.info("🎨 Loading VAE...")
            vae = wan_vae_loader.loadmodel(os.path.basename(MODEL_CONFIGS["wan21_vae"]), "fp16")[0]
            
            logger.info("🔄 Creating image embeddings...")
            image_embeds = multitalk_img2vid.process(
                vae, width, height, frames, fps, 
                False,  # use_first_frame_only
                'mkl',  # method
                loaded_image, 
                False,  # force_cpu
                clip_vision_output
            )[0]
            
            # STEP 6: Load Talk model (MultiTalk hoặc InfiniteTalk)
            if mode == "multitalk":
                logger.info("🗣️ Loading MultiTalk Model...")
                talk_model_path = MODEL_CONFIGS["multitalk_model"]
            else:
                logger.info("🗣️ Loading InfiniteTalk Model...")
                talk_model_path = MODEL_CONFIGS["infinitalk_model"]
            
            multitalk_model = multitalk_loader.loadmodel(os.path.basename(talk_model_path), "fp16")[0]
            
            # STEP 7: Setup LoRA nếu enabled
            wan_speed_lora = None
            if use_speed_lora:
                logger.info("⚡ Loading Speed LoRA...")
                wan_speed_lora = wan_lora_select.getlorapath(
                    os.path.basename(MODEL_CONFIGS["speed_lora"]), 
                    speed_lora_strength, 
                    None, {}, None, False, False
                )[0]
            
            # STEP 8: Setup block swap nếu enabled
            block_swap_args = None
            if use_block_swap:
                logger.info(f"🔄 Setting up block swap (blocks: {blocks_to_swap})...")
                block_swap_args = block_swapper.setargs(
                    blocks_to_swap=blocks_to_swap,
                    offload_img_emb=False,
                    offload_txt_emb=False,
                    use_non_blocking=True,
                    vace_blocks_to_swap=0,
                    prefetch_blocks=0,
                    block_swap_debug=False
                )[0]
            
            # STEP 9: Load WAN2.1 base model với talk model integration
            logger.info("🎯 Loading WAN2.1 base model...")
            model = wan_model_loader.loadmodel(
                model=os.path.basename(MODEL_CONFIGS["wan21_model"]),
                base_precision="fp16",
                load_device="offload_device",
                quantization="disabled",
                compile_args=None,
                attention_mode="sageattn",
                block_swap_args=block_swap_args,
                lora=wan_speed_lora,
                vram_management_args=None,
                vace_model=None,
                fantasytalking_model=None,
                multitalk_model=multitalk_model,
                fantasyportrait_model=None
            )[0]
            
            del multitalk_model
            clear_memory()
            
            # STEP 10: Sample với WanVideoSampler (EXACT theo notebook)
            logger.info(f"🎬 Sampling video...")
            sampled = wan_sampler.process(
                model=model,
                image_embeds=image_embeds,
                shift=8,  # Flow shift theo notebook
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
                context_options=None,
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
                multitalk_embeds=wav2vec_embeds,  # KEY: audio embeddings
                freeinit_args=None,
                start_step=0,
                end_step=-1,
                add_noise_to_samples=False
            )[0]
            
            del model
            clear_memory()
            
            # STEP 11: Decode latents to video frames
            logger.info("🎨 Decoding latents...")
            decoded = wan_vae_decoder.decode(
                vae=vae,
                samples=sampled,
                enable_vae_tiling=False,
                tile_x=272,
                tile_y=272,
                tile_stride_x=144,
                tile_stride_y=128,
                normalization="default"
            )[0]
            
            del vae
            clear_memory()
            
            # STEP 12: Save video với audio
            logger.info("💾 Saving final video với audio...")
            output_path = f"/app/ComfyUI/output/wan21_{mode}_{uuid.uuid4().hex[:8]}.mp4"
            
            final_output_path = save_video_with_audio(
                decoded, output_path, fps, output_audio_path
            )
            
            logger.info(f"✅ Video generation completed: {final_output_path}")
            return final_output_path
            
    except Exception as e:
        logger.error(f"❌ WAN2.1 video generation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
        clear_memory()

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file to MinIO storage"""
    try:
        if not minio_client:
            raise RuntimeError("MinIO client not initialized")
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"📤 Uploading to MinIO: {object_name} ({file_size_mb:.1f}MB)")
        
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        file_url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        
        logger.info(f"✅ Upload completed: {file_url}")
        return file_url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise e

def validate_input_parameters(job_input: dict) -> tuple[bool, str]:
    """Validate input parameters for WAN2.1 workflow"""
    try:
        # Required parameters
        required_params = ["image_url", "audio_urls"]
        for param in required_params:
            if param not in job_input or not job_input[param]:
                return False, f"Missing required parameter: {param}"
        
        # Validate image URL
        image_url = job_input["image_url"]
        try:
            response = requests.head(image_url, timeout=10)
            if response.status_code != 200:
                return False, f"Image URL not accessible: {response.status_code}"
        except Exception as e:
            return False, f"Image URL validation failed: {str(e)}"
        
        # Validate audio URLs
        audio_urls = job_input["audio_urls"]
        if not isinstance(audio_urls, list) or len(audio_urls) == 0:
            return False, "audio_urls must be a non-empty list"
        
        if len(audio_urls) > 4:
            return False, "Maximum 4 audio files supported"
        
        # Validate dimensions
        width = job_input.get("width", 400)
        height = job_input.get("height", 704)
        if not (256 <= width <= 1536 and 256 <= height <= 1536):
            return False, "Width and height must be between 256 and 1536"
        
        # Validate mode
        mode = job_input.get("mode", "multitalk")
        if mode not in ["multitalk", "infinitetalk"]:
            return False, "Mode must be 'multitalk' or 'infinitetalk'"
        
        # Validate steps
        steps = job_input.get("steps", 4)
        if not (1 <= steps <= 50):
            return False, "Steps must be between 1 and 50"
        
        # Validate CFG scale
        cfg_scale = job_input.get("cfg_scale", 1.0)
        if not (0.1 <= cfg_scale <= 20.0):
            return False, "CFG scale must be between 0.1 and 20.0"
        
        # Validate FPS
        fps = job_input.get("fps", 25)
        if not (10 <= fps <= 60):
            return False, "FPS must be between 10 and 60"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Parameter validation error: {str(e)}"

def handler(job):
    """
    MAIN RunPod handler for WAN2.1 InfiniteTalk/MultiTalk
    Complete workflow implementation với comprehensive error handling
    """
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        
        # Validate input parameters
        is_valid, validation_message = validate_input_parameters(job_input)
        if not is_valid:
            return {
                "error": validation_message,
                "status": "failed",
                "job_id": job_id
            }
        
        # Extract validated parameters
        image_url = job_input["image_url"]
        audio_urls = job_input["audio_urls"]
        
        # Extract parameters với defaults từ notebook
        parameters = {
            "positive_prompt": job_input.get("positive_prompt", "The person takes turns talking"),
            "negative_prompt": job_input.get("negative_prompt", "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality"),
            "width": job_input.get("width", 400),
            "height": job_input.get("height", 704),
            "fps": job_input.get("fps", 25),
            "seed": job_input.get("seed", 0),
            "steps": job_input.get("steps", 4),
            "cfg_scale": job_input.get("cfg_scale", 1.0),
            "scheduler": job_input.get("scheduler", "flowmatch_distill"),
            "mode": job_input.get("mode", "multitalk"),
            "use_block_swap": job_input.get("use_block_swap", False),
            "blocks_to_swap": job_input.get("blocks_to_swap", 20),
            "use_speed_lora": job_input.get("use_speed_lora", True),
            "speed_lora_strength": job_input.get("speed_lora_strength", 1.0),
            "max_audio_duration": job_input.get("max_audio_duration", None)
        }
        
        logger.info(f"🚀 Job {job_id}: WAN2.1 {parameters['mode'].upper()} Generation Started")
        logger.info(f"🖼️ Image: {image_url}")
        logger.info(f"🎵 Audio files: {len(audio_urls)}")
        logger.info(f"📝 Prompt: {parameters['positive_prompt'][:100]}...")
        logger.info(f"⚙️ Resolution: {parameters['width']}x{parameters['height']} @ {parameters['fps']}fps")
        
        # Verify models for selected mode
        models_ok, missing_models = verify_models(parameters['mode'])
        if not models_ok:
            return {
                "error": "Required models are missing",
                "missing_models": missing_models,
                "status": "failed",
                "mode": parameters['mode']
            }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download input image
            image_path = os.path.join(temp_dir, "input_image.jpg")
            logger.info("📥 Downloading input image...")
            try:
                response = requests.get(image_url, timeout=60, stream=True)
                response.raise_for_status()
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                image_size_mb = os.path.getsize(image_path) / (1024 * 1024)
                logger.info(f"✅ Image downloaded: {image_size_mb:.1f}MB")
            except Exception as e:
                return {"error": f"Failed to download image: {str(e)}"}
            
            # Download audio files
            audio_paths = []
            for i, audio_url in enumerate(audio_urls):
                if audio_url:
                    audio_path = os.path.join(temp_dir, f"audio_{i}.wav")
                    logger.info(f"📥 Downloading audio {i+1}/{len(audio_urls)}...")
                    try:
                        response = requests.get(audio_url, timeout=60, stream=True)
                        response.raise_for_status()
                        with open(audio_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                        audio_paths.append(audio_path)
                        logger.info(f"✅ Audio {i+1} downloaded: {audio_size_mb:.1f}MB")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to download audio {i+1}: {e}")
                        audio_paths.append(None)
                else:
                    audio_paths.append(None)
            
            # Verify at least one audio file was downloaded
            valid_audio_paths = [p for p in audio_paths if p is not None]
            if not valid_audio_paths:
                return {"error": "No valid audio files could be downloaded"}
            
            # Generate video với COMPLETE WAN2.1 workflow
            logger.info("🎬 Starting WAN2.1 video generation...")
            generation_start = time.time()
            
            output_path = generate_video_wan21_complete(
                image_path=image_path,
                audio_paths=audio_paths,
                **parameters
            )
            
            generation_time = time.time() - generation_start
            
            if not output_path or not os.path.exists(output_path):
                return {"error": "Video generation failed"}
            
            # Upload result to MinIO
            logger.info("📤 Uploading result to storage...")
            output_filename = f"wan21_{parameters['mode']}_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            try:
                output_url = upload_to_minio(output_path, output_filename)
            except Exception as e:
                return {"error": f"Failed to upload result: {str(e)}"}
            
            # Calculate statistics
            total_time = time.time() - start_time
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            
            # Determine actual seed used
            actual_seed = parameters["seed"] if parameters["seed"] != 0 else "auto-generated"
            
            logger.info(f"✅ Job {job_id} completed successfully!")
            logger.info(f"⏱️ Total time: {total_time:.1f}s (generation: {generation_time:.1f}s)")
            logger.info(f"📊 Output: {file_size_mb:.1f}MB")
            
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
                    "negative_prompt": parameters["negative_prompt"][:100] + "..." if len(parameters["negative_prompt"]) > 100 else parameters["negative_prompt"],
                    "steps": parameters["steps"],
                    "cfg_scale": parameters["cfg_scale"],
                    "seed": actual_seed,
                    "scheduler": parameters["scheduler"],
                    "mode": parameters["mode"],
                    "block_swap": parameters["use_block_swap"],
                    "speed_lora": parameters["use_speed_lora"],
                    "workflow_version": "WAN21_INFINITETALK_MULTITALK_COMPLETE"
                },
                "status": "completed"
            }
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Handler error for job {job_id}: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "error": error_msg,
            "status": "failed",
            "processing_time_seconds": round(time.time() - start_time, 2),
            "job_id": job_id
        }
    finally:
        clear_memory()

def health_check():
    """Health check function for WAN2.1 service"""
    try:
        # Check CUDA
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        # Check WanVideoWrapper modules
        if not WANVIDEO_AVAILABLE:
            return False, "WanVideoWrapper not available"
        
        # Check models for both modes
        multitalk_ok, _ = verify_models("multitalk")
        infinitetalk_ok, _ = verify_models("infinitetalk")
        
        if not (multitalk_ok or infinitetalk_ok):
            return False, "No complete model sets available"
        
        # Check MinIO
        if not minio_client:
            return False, "MinIO not available"
        
        available_modes = []
        if multitalk_ok:
            available_modes.append("multitalk")
        if infinitetalk_ok:
            available_modes.append("infinitetalk")
        
        return True, f"Ready - Modes: {', '.join(available_modes)}"
        
    except Exception as e:
        return False, f"Health check failed: {str(e)}"

if __name__ == "__main__":
    logger.info("🚀 Starting WAN2.1 InfiniteTalk/MultiTalk Serverless Worker...")
    logger.info(f"🔥 PyTorch: {torch.__version__}")
    logger.info(f"🎯 CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"💾 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    try:
        # Health check on startup
        health_ok, health_msg = health_check()
        if not health_ok:
            logger.error(f"❌ Health check failed: {health_msg}")
            sys.exit(1)
        
        logger.info(f"✅ Health check passed: {health_msg}")
        logger.info("🎬 Ready to process WAN2.1 InfiniteTalk/MultiTalk requests...")
        logger.info("🔧 Complete workflow implementation with audio-driven video generation!")
        
        # Start RunPod worker
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
