from faster_whisper import WhisperModel
from datetime import datetime
import soundfile as sf
import numpy as np

# モデルのロード（smallモデルを例示。必要に応じて変更可）
model = WhisperModel("small", device="cpu", compute_type="int8")

# 音声ファイルのパス
input_audio = "rec_threading.wav"

# 音声ファイルをNumPy配列として読み込む
waveform, samplerate = sf.read(input_audio)

print(waveform.shape, samplerate)
print(waveform.dtype)
# 音声データの形状を確認

# 文字起こしの実行
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "文字起こしを開始")
segments, info = model.transcribe(waveform, without_timestamps=True, language="ja", beam_size=5)
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "文字起こしが完了")
print("音声ファイルの長さ:", info.duration, "秒")


# 結果をテキストファイルに保存
with open("output.txt", "w", encoding="utf-8") as f:
    for segment in segments:
        f.write(segment.text + "\n")

print("文字起こしが完了し、output.txtに保存されました。")