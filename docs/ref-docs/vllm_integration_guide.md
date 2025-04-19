# vLLM による Qwen2.5-Omni 統合ガイド

このドキュメントでは、vLLMを使用してQwen2.5-Omniモデルを統合するための方法について説明します。

## 概要

Qwen2.5-Omniは、テキスト、画像、音声、ビデオなどの多様なモダリティを認識し、テキスト生成と自然な音声合成を同時に行うことができるエンドツーエンドのマルチモーダルモデルです。vLLMは、高速な推論と効率的なリソース管理を提供するサービングフレームワークです。

## vLLMのインストール

Qwen2.5-Omniをサポートするvllmをインストールするには、特定のブランチからインストールする必要があります：

```bash
git clone -b qwen2_omni_public https://github.com/fyabc/vllm.git
cd vllm
git checkout 729feed3ec2beefe63fda30a345ef363d08062f8
pip install setuptools_scm torchdiffeq resampy x_transformers qwen-omni-utils accelerate
pip install -r requirements/cuda.txt 
pip install .
pip install git+https://github.com/huggingface/transformers
```

## ローカル推論

vLLMを使用してQwen2.5-Omniをローカルで推論するには、以下のコマンドを使用します：

### テキスト出力のみ（単一GPU）

```bash
python end2end.py --model Qwen/Qwen2.5-Omni-7B --prompt audio-in-video-v2 --enforce-eager --thinker-only
```

### テキスト出力のみ（複数GPU）

```bash
python end2end.py --model Qwen/Qwen2.5-Omni-7B --prompt audio-in-video-v2 --enforce-eager --thinker-only --thinker-devices [0,1,2,3] --thinker-gpu-memory-utilization 0.9 
```

### 音声出力（単一GPU）

```bash
python end2end.py --model Qwen/Qwen2.5-Omni-7B --prompt audio-in-video-v2 --enforce-eager --do-wave --voice-type Chelsie --warmup-voice-type Chelsie --output-dir output_wav
```

### 音声出力（複数GPU）

```bash
python end2end.py --model Qwen/Qwen2.5-Omni-7B --prompt audio-in-video-v2 --enforce-eager --do-wave --voice-type Chelsie --warmup-voice-type Chelsie --thinker-devices [0,1] --talker-devices [2] --code2wav-devices [3] --thinker-gpu-memory-utilization 0.9 --talker-gpu-memory-utilization 0.9 --output-dir output_wav
```

## APIサービング

vLLMを使用してQwen2.5-Omniをサービングするには、以下のコマンドを使用します：

### 単一GPU

```bash
VLLM_USE_V1=0 vllm serve /path/to/Qwen2.5-Omni-7B/ --port 8000 --host 127.0.0.1 --dtype bfloat16
```

### 複数GPU

```bash
VLLM_USE_V1=0 vllm serve /path/to/Qwen2.5-Omni-7B/ --port 8000 --host 127.0.0.1 --dtype bfloat16 -tp 4
```

## APIの使用例

サービングしたAPIを使用する例：

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/cough.wav"}},
        {"type": "text", "text": "What is the text in the illustrate ans what it the sound in the audio?"}
    ]}
    ]
    }'
```

## Dockerを使用した展開

簡単に展開するために、事前構築された環境を持つDockerイメージが提供されています：

```bash
docker run --gpus all --ipc=host --network=host --rm --name qwen2.5-omni -it qwenllm/qwen-omni:2.5-cu121 bash
```

Webデモを起動するには：

```bash
bash docker/docker_web_demo.sh --checkpoint /path/to/Qwen2.5-Omni-7B
```

FlashAttention-2を有効にするには：

```bash
bash docker/docker_web_demo.sh --checkpoint /path/to/Qwen2.5-Omni-7B --flash-attn2
```

## 注意事項

- vLLMサービングは現在、Qwen2.5-Omniのテキスト出力のみをサポートしています。
- 音声出力を含む完全な機能を使用するには、ローカル推論またはTransformersを使用してください。
- GPUメモリの使用量を最適化するために、`--gpu-memory-utilization`パラメータを調整できます。
- 長いコンテキストを処理する場合は、`--max-model-len`パラメータを調整して、OOMの問題を解決できます。
