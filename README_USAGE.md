# WAN2.1 InfiniteTalk/MultiTalk RunPod Serverless

Dịch vụ RunPod Serverless cho audio-driven video generation sử dụng WAN2.1 với InfiniteTalk/MultiTalk models.

## 🚀 Tính Năng

- **Audio-Driven Video Generation**: Tạo video từ hình ảnh và audio với lip-sync chính xác
- **Multi-Speaker Support**: Hỗ trợ tối đa 4 audio files cho conversation scenarios  
- **Dual Modes**: 
  - `multitalk`: Một model cho cả single/multiple speakers
  - `infinitetalk`: Models riêng cho single/multiple speakers
- **High Quality**: WAN2.1 14B parameters với Q4_K_M quantization
- **Speed Optimization**: LightX2V LoRA cho tăng tốc inference
- **Flexible Resolution**: Hỗ trợ nhiều kích thước khác nhau

## 🏗️ Build và Deploy

### 1. Build Docker Image

```bash
# Build image
docker build -t wan21-infinitetalk:latest -f Dockerfile .

# Hoặc với debug info
chmod +x build_wan22_debug.sh
./build_wan22_debug.sh
```

### 2. Deploy trên RunPod

```bash
# Push to registry
docker tag wan21-infinitetalk:latest your-registry/wan21-infinitetalk:latest
docker push your-registry/wan21-infinitetalk:latest

# Deploy trên RunPod với image URL này
```

## 📋 API Usage

### Request Format

```json
{
  "input": {
    "image_url": "https://example.com/image.jpg",
    "audio_urls": [
      "https://example.com/audio1.wav",
      "https://example.com/audio2.wav"
    ],
    "positive_prompt": "The woman and man take turns talking to each other",
    "negative_prompt": "low quality, static, deformed",
    "width": 400,
    "height": 704,
    "fps": 25,
    "steps": 4,
    "cfg_scale": 1.0,
    "scheduler": "flowmatch_distill",
    "mode": "multitalk",
    "use_block_swap": false,
    "blocks_to_swap": 20,
    "use_speed_lora": true,
    "speed_lora_strength": 1.0,
    "max_audio_duration": 30,
    "seed": 0
  }
}
```

### Parameters

#### Required
- `image_url`: URL của hình ảnh đầu vào (JPG/PNG)
- `audio_urls`: List các URL audio files (tối đa 4, WAV/MP3)

#### Optional Video Settings
- `positive_prompt`: Mô tả video muốn tạo (default: "The person takes turns talking")
- `negative_prompt`: Những gì muốn tránh
- `width`: Chiều rộng video (256-1536, default: 400)
- `height`: Chiều cao video (256-1536, default: 704)
- `fps`: Frames per second (10-60, default: 25)

#### Optional Generation Settings
- `steps`: Số steps sampling (1-50, default: 4)
- `cfg_scale`: Classifier-free guidance scale (0.1-20.0, default: 1.0)
- `scheduler`: Sampling scheduler (default: "flowmatch_distill")
- `seed`: Random seed (default: 0 = auto)

#### Optional Advanced Settings
- `mode`: "multitalk" hoặc "infinitetalk" (default: "multitalk")
- `use_block_swap`: Enable block swapping để tiết kiệm VRAM
- `blocks_to_swap`: Số blocks để swap (default: 20)
- `use_speed_lora`: Enable speed LoRA (default: true)
- `speed_lora_strength`: Strength của speed LoRA (default: 1.0)
- `max_audio_duration`: Giới hạn độ dài audio (seconds)

### Response Format

```json
{
  "output_video_url": "https://media.aiclip.ai/video/wan21_multitalk_xxx.mp4",
  "processing_time_seconds": 45.2,
  "generation_time_seconds": 38.7,
  "video_info": {
    "width": 400,
    "height": 704,
    "fps": 25,
    "file_size_mb": 12.5,
    "mode": "multitalk",
    "audio_files_used": 2
  },
  "generation_params": {
    "positive_prompt": "The woman and man take turns talking to each other",
    "steps": 4,
    "cfg_scale": 1.0,
    "seed": "auto-generated",
    "scheduler": "flowmatch_distill",
    "mode": "multitalk",
    "block_swap": false,
    "speed_lora": true,
    "workflow_version": "WAN21_INFINITETALK_MULTITALK_COMPLETE"
  },
  "status": "completed"
}
```

## 🎯 Use Cases

### 1. Single Speaker (Monologue)
```json
{
  "input": {
    "image_url": "https://example.com/person.jpg",
    "audio_urls": ["https://example.com/speech.wav"],
    "positive_prompt": "A person speaking naturally",
    "mode": "infinitetalk"
  }
}
```

### 2. Conversation (Multiple Speakers)
```json
{
  "input": {
    "image_url": "https://example.com/two_people.jpg", 
    "audio_urls": [
      "https://example.com/speaker1.wav",
      "https://example.com/speaker2.wav"
    ],
    "positive_prompt": "Two people having a conversation",
    "mode": "multitalk"
  }
}
```

### 3. High Quality với Block Swap
```json
{
  "input": {
    "image_url": "https://example.com/image.jpg",
    "audio_urls": ["https://example.com/long_audio.wav"],
    "width": 720,
    "height": 1280,
    "max_audio_duration": 60,
    "use_block_swap": true,
    "blocks_to_swap": 15
  }
}
```

## ⚡ Performance Tips

### 1. Optimal Settings cho Speed
```json
{
  "steps": 4,
  "use_speed_lora": true,
  "speed_lora_strength": 1.0,
  "scheduler": "flowmatch_distill"
}
```

### 2. Memory Management cho Long Videos
```json
{
  "use_block_swap": true,
  "blocks_to_swap": 20,
  "max_audio_duration": 30
}
```

### 3. Quality vs Speed Trade-offs
- **Fastest**: `steps: 4, use_speed_lora: true`
- **Balanced**: `steps: 6, use_speed_lora: true` 
- **Quality**: `steps: 8, use_speed_lora: false`

## 🔧 Model Details

### Pre-loaded Models
- **WAN2.1 Base**: wan2.1-i2v-14b-480p-Q4_K_M.gguf (14B parameters)
- **MultiTalk**: WanVideo_2_1_Multitalk_14B_fp8_e4m3fn.safetensors
- **InfiniteTalk**: Wan2_1-InfiniteTalk-Multi_fp16.safetensors
- **Speed LoRA**: lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors
- **VAE**: wan_2.1_vae.safetensors
- **Text Encoder**: umt5_xxl_fp8_e4m3fn_scaled.safetensors
- **CLIP Vision**: clip_vision_h.safetensors
- **Wav2Vec**: chinese-wav2vec2-base-fairseq-ckpt.pt

### System Requirements
- **GPU**: Minimum 16GB VRAM (24GB recommended cho long videos)
- **RAM**: 32GB recommended
- **Storage**: ~50GB cho all models

## 🐛 Troubleshooting

### Common Issues

1. **OOM Errors với Long Audio**
   ```json
   {
     "use_block_swap": true,
     "blocks_to_swap": 25,
     "max_audio_duration": 20
   }
   ```

2. **Poor Lip Sync Quality**
   - Sử dụng audio chất lượng cao (44.1kHz+)
   - Đảm bảo audio rõ ràng không có noise
   - Thử mode "infinitetalk" cho single speaker

3. **Slow Generation**
   ```json
   {
     "steps": 4,
     "use_speed_lora": true,
     "scheduler": "flowmatch_distill"
   }
   ```

### Error Codes
- `Missing required parameter`: Thiếu image_url hoặc audio_urls
- `Required models are missing`: Models chưa được tải đầy đủ
- `No valid audio files could be downloaded`: Không tải được audio nào
- `Video generation failed`: Lỗi trong quá trình generation

## 📊 Benchmarks

| Resolution | Audio Duration | Generation Time | VRAM Usage |
|------------|----------------|-----------------|------------|
| 400x704    | 10s            | ~30s           | ~12GB      |
| 720x1280   | 10s            | ~45s           | ~16GB      |
| 400x704    | 30s            | ~80s           | ~14GB      |
| 720x1280   | 30s (block swap) | ~120s        | ~18GB      |

*Results trên RTX 4090 24GB*

## 🔗 Related

- [WAN2.1 Original](https://github.com/alibaba/Wan_Video)
- [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk)  
- [MultiTalk](https://github.com/MeiGen-AI/MultiTalk)
- [ComfyUI WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)

## 📄 License

Model weights và code được distribute theo respective licenses của original authors.
