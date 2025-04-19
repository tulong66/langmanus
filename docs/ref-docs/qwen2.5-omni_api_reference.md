# Qwen2.5-Omni API リファレンス

このドキュメントでは、Qwen2.5-Omniモデルを使用するためのAPIリファレンスを提供します。

## モデル概要

Qwen2.5-Omniは、テキスト、画像、音声、ビデオなどの多様なモダリティを認識し、テキスト生成と自然な音声合成を同時に行うことができるエンドツーエンドのマルチモーダルモデルです。

## Transformers API

### モデルの読み込み

```python
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# デフォルト: 利用可能なデバイスにモデルを読み込む
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B", 
    torch_dtype="auto", 
    device_map="auto"
)

# Flash Attention 2を有効にする場合（推奨）
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
```

### 会話形式

```python
conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "path/to/video.mp4"},
        ],
    },
]
```

### 推論

```python
# 音声付きビデオを使用するかどうか
USE_AUDIO_IN_VIDEO = True

# 推論の準備
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# 出力テキストと音声の生成
text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)
```

## 音声タイプの変更

Qwen2.5-Omniは、出力音声の声を変更する機能をサポートしています。`"Qwen/Qwen2.5-Omni-7B"`チェックポイントは、以下の2つの声タイプをサポートしています：

| 声タイプ | 性別 | 説明 |
|---------|------|------|
| Chelsie | 女性 | 優しい温かみと明るさを持つ、蜜のような、ビロードのような声 |
| Ethan | 男性 | 明るく、元気で、温かみがあり、親しみやすい雰囲気を持つ声 |

`generate`関数の`speaker`パラメータを使用して、声のタイプを指定できます。デフォルトでは、`speaker`が指定されていない場合、デフォルトの声タイプは`Chelsie`です。

```python
text_ids, audio = model.generate(**inputs, speaker="Chelsie")
text_ids, audio = model.generate(**inputs, speaker="Ethan")
```

## 音声出力の無効化

モデルはテキストと音声の両方の出力をサポートしていますが、音声出力が不要な場合は、モデルの初期化後に`model.disable_talker()`を呼び出すことができます。このオプションを使用すると、GPUメモリを約`2GB`節約できますが、`generate`関数の`return_audio`オプションは`False`にのみ設定できます。

```python
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto"
)
model.disable_talker()
```

より柔軟な体験を得るために、`generate`関数を呼び出すときに`return_audio`を設定して、音声を返すかどうかを決定することもできます。`return_audio`が`False`に設定されている場合、モデルはテキスト応答をより速く取得するためにテキスト出力のみを返します。

```python
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto"
)
...
text_ids = model.generate(**inputs, return_audio=False)
```

## バッチ推論

モデルは、`return_audio=False`が設定されている場合、テキスト、画像、音声、ビデオなどの様々なタイプの混合サンプルで構成される入力をバッチ処理できます。

```python
# バッチ推論のためのサンプルメッセージ

# ビデオのみの会話
conversation1 = [...]

# 音声のみの会話
conversation2 = [...]

# 純粋なテキストの会話
conversation3 = [...]

# 混合メディアの会話
conversation4 = [...]

# バッチ処理のためのメッセージの組み合わせ
conversations = [conversation1, conversation2, conversation3, conversation4]

# ビデオ内の音声を使用するかどうか
USE_AUDIO_IN_VIDEO = True

# バッチ推論の準備
text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)

inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# バッチ推論
text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
```

## OpenAI互換API

Qwen2.5-Omniは、OpenAI互換APIを通じてアクセスすることもできます。

```python
import base64
import numpy as np
import soundfile as sf

from openai import OpenAI

client = OpenAI(
    api_key="your_api_key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

messages = [
    {
        "role": "system",
        "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    },
    {
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"},
        ],
    },
]

# Qwen-Omniはストリームモードのみをサポート
completion = client.chat.completions.create(
    model="qwen-omni-turbo",
    messages=messages,
    modalities=["text", "audio"],
    audio={
        "voice": "Cherry", # Cherry, Ethan, Serena, Chelsieが利用可能
        "format": "wav"
    },
    stream=True,
    stream_options={"include_usage": True}
)

text = []
audio_string = ""
for chunk in completion:
    if chunk.choices:
        if hasattr(chunk.choices[0].delta, "audio"):
            try:
                audio_string += chunk.choices[0].delta.audio["data"]
            except Exception as e:
                text.append(chunk.choices[0].delta.audio["transcript"])
    else:
        print(chunk.usage)

print("".join(text))
wav_bytes = base64.b64decode(audio_string)
wav_array = np.frombuffer(wav_bytes, dtype=np.int16)
sf.write("output.wav", wav_array, samplerate=24000)
```

## カスタマイズ設定

Qwen2.5-Omniは、[音声出力](#音声出力の無効化)を使用する場合（ローカル展開とAPI推論の両方を含む）、プロンプト設定をサポートしていません。モデルの出力を制御したり、モデルのパーソナリティ設定を変更したりする必要がある場合は、以下のように会話テンプレートに類似のコンテンツを追加することをお勧めします：

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "You are a shopping guide, now responsible for introducing various products."},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Sure, I got it."},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Who are you?"},
        ],
    },
]
```
