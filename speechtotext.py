from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import threading
import queue
import time
import sys
import threading


class WhisperTranscriber:
    SILENCE_THRESHOLD = 5e-3  # 無音判定の閾値

    def __init__(self, model_size='base', language='ja'):
        """
        Whisper音声認識クラス
        """
        self.model = WhisperModel(model_size, device="cpu")
        
        # 音声入力設定
        self.pyaudio = pyaudio.PyAudio()
        self.sample_rate = 16000
        self.chunk_duration = 3
        self.chunks_per_inference = int(self.sample_rate * self.chunk_duration)
        
        # スレッディング用キュー
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        self.language = language

    def transcribe_audio(self):
        """
        リアルタイム音声文字起こし
        """
        if self.model is None:
            print("モデルの初期化に失敗しました")
            return

        audio_data = []
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=1)
                audio_data.extend(chunk)

                if len(audio_data) >= self.chunks_per_inference:
                    # テンソル変換
                    audio_np = np.array(audio_data[:self.chunks_per_inference])
                    
                    # 無音判定
                    if np.abs(audio_np).mean() < self.SILENCE_THRESHOLD:
                        # 無音の場合はスキップ
                        audio_data = audio_data[self.chunks_per_inference:]
                        continue
                    
                    # 推論を実行
                    results, info = self.model.transcribe(
                        audio_np, 
                        language=self.language
                    )
                    
                    for result in results:
                        print(f"文字起こし結果: {result.text}")
                        with open("output.txt", "a", encoding="utf-8") as f:
                            f.write(result.text + "\n")
                    audio_data = audio_data[self.chunks_per_inference:]

            except queue.Empty:
                continue
            except Exception as e:
                print(f"推論中にエラー: {e}")

    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        音声データ処理用コールバック
        """
        audio_chunk = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_chunk)
        return (None, pyaudio.paContinue)

    def start_recording(self):
        """
        音声録音開始
        """
        self.stream = self.pyaudio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunks_per_inference,
            stream_callback=self.audio_callback,
        )
        print("音声録音を開始")

    def keyboard_listener(self):
        """
        Qキーで終了するリスナー
        """
        if sys.platform == 'win32':
            import msvcrt
            print("Qキーを押すと終了します。")
            while not self.stop_event.is_set():
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key in [b'q', b'Q']:
                        print("Qキーが押されました。終了します。")
                        self.stop_event.set()
                        break

    def run(self):
        """
        音声認識プロセス実行
        """
        self.start_recording()
        transcribe_thread = threading.Thread(target=self.transcribe_audio)
        transcribe_thread.start()
        keyboard_thread = threading.Thread(target=self.keyboard_listener)
        keyboard_thread.daemon = True
        keyboard_thread.start()
        transcribe_thread.join()
        self.stop_event.set()
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()

def main():
    transcriber = WhisperTranscriber(
        model_size='small',  
        language='ja'
    )
    transcriber.run()

if __name__ == "__main__":
    main()