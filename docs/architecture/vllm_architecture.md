# vLLMベースのQwen2.5-Omni統合アーキテクチャ

## 概要

このドキュメントでは、vLLMを使用したQwen2.5-Omniモデルの統合アーキテクチャについて説明します。vLLMは、PagedAttentionなどの最適化技術を使用して、高速な推論と効率的なリソース管理を提供するサービングフレームワークです。

## アーキテクチャ図

```
┌─────────────────┐      ┌───────────────────┐      ┌─────────────────┐
│                 │      │                   │      │                 │
│  ユーザーインターフェース │ ──→ │  vLLM サーバー     │ ──→ │  Qwen2.5-Omni  │
│                 │      │                   │      │                 │
└─────────────────┘      └───────────────────┘      └─────────────────┘
                                   │
                                   ↓
                         ┌───────────────────┐
                         │                   │
                         │  音声生成モジュール   │
                         │  (オプション)        │
                         │                   │
                         └───────────────────┘
```

## コンポーネント詳細

### 1. vLLMサーバー

- **役割**: Qwen2.5-Omniモデルのサービング
- **主要機能**:
  - PagedAttentionによる効率的なKVキャッシュ管理
  - 連続バッチ処理によるスループットの最適化
  - テンソル並列処理によるマルチGPUサポート
  - OpenAI互換APIの提供

### 2. Qwen2.5-Omniモデル

- **役割**: マルチモーダル入力の処理とテキスト生成
- **主要コンポーネント**:
  - Thinker: テキスト生成を担当するLLMコンポーネント
  - 視覚エンコーダー: 画像とビデオの処理
  - 音声エンコーダー: 音声入力の処理
  - TMRoPE (Time-aligned Multimodal RoPE): ビデオ入力と音声のタイムスタンプを同期

### 3. 音声生成モジュール（オプション）

- **役割**: テキスト出力から音声を生成
- **主要コンポーネント**:
  - Talker: 音声生成を担当するコンポーネント
  - Code2Wav: 音声コードから波形を生成

## データフロー

1. ユーザーがマルチモーダル入力（テキスト、画像、音声、ビデオ）を提供
2. vLLMサーバーが入力を前処理し、Qwen2.5-Omniモデルに送信
3. Qwen2.5-Omniモデルが入力を処理し、テキスト出力を生成
4. （オプション）音声生成モジュールがテキスト出力から音声を生成
5. 結果がユーザーに返される

## デプロイメントオプション

### 1. テキスト出力のみ（Thinkerのみ）

```bash
VLLM_USE_V1=0 vllm serve /path/to/Qwen2.5-Omni-7B/ --port 8000 --host 127.0.0.1 --dtype bfloat16
```

### 2. テキストと音声出力（Thinker + Talker + Code2Wav）

```bash
python end2end.py --model Qwen/Qwen2.5-Omni-7B --prompt audio-in-video-v2 --enforce-eager --do-wave --voice-type Chelsie --warmup-voice-type Chelsie --output-dir output_wav
```

### 3. マルチGPUデプロイメント

```bash
# テキスト出力のみ
VLLM_USE_V1=0 vllm serve /path/to/Qwen2.5-Omni-7B/ --port 8000 --host 127.0.0.1 --dtype bfloat16 -tp 4

# テキストと音声出力
python end2end.py --model Qwen/Qwen2.5-Omni-7B --prompt audio-in-video-v2 --enforce-eager --do-wave --voice-type Chelsie --warmup-voice-type Chelsie --thinker-devices [0,1] --talker-devices [2] --code2wav-devices [3] --thinker-gpu-memory-utilization 0.9 --talker-gpu-memory-utilization 0.9 --output-dir output_wav
```

## メモリ要件

| 精度 | 15秒ビデオ | 30秒ビデオ | 60秒ビデオ |
|------|-----------|-----------|-----------|
| FP32 | 93.56 GB  | 非推奨     | 非推奨     |
| BF16 | 31.11 GB  | 41.85 GB  | 60.19 GB  |

注: 上記の表は、`transformers`と`BF16`（`attn_implementation="flash_attention_2"`でテスト）を使用した推論の理論的な最小メモリ要件を示していますが、実際のメモリ使用量は通常、少なくとも1.2倍高くなります。

## 最適化のヒント

1. **GPUメモリの使用量**:
   - `--gpu-memory-utilization`パラメータを調整して、GPUメモリの使用量を制御できます
   - デフォルトは0.9で、これはvLLMがGPUメモリの90%を事前に割り当てることを意味します

2. **モデル長の制限**:
   - `--max-model-len`パラメータを調整して、OOMの問題を解決できます
   - デフォルトは32,768トークンで、これはより多くのメモリを必要とします

3. **KVキャッシュの量子化**:
   - FP8 E5M2 KVキャッシュを使用して、メモリ使用量を削減できます
   - `--kv-cache-dtype fp8_e5m2`パラメータを追加します

4. **テンソル並列処理**:
   - 複数のGPUにモデルを分散させるには、`-tp`パラメータを使用します
   - 例: `-tp 4`は4つのGPUにモデルを分散させます
