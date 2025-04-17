# Whisper Realtime Speech-to-Text

Whisper Realtime Speech-to-Textは、OpenAIのWhisperモデルを利用したリアルタイム音声認識ツールです。マイクからの音声をリアルタイムでテキスト化し、無音検出やQキーによる終了操作に対応しています。

## 特徴
- **リアルタイム音声認識**: マイク入力を即座にWhisperで文字起こし
- **無音検出**: 一定時間無音が続くと自動で文字起こしを実行
- **Qキーで終了**: Qキーを押すことで安全に録音・認識を終了
- **テキスト出力**: 認識結果は`output.txt`に追記保存

## 必要環境
- Python 3.8以上
- Windows（他OSの場合はキーボード終了処理を調整してください）

## インストール
1. リポジトリをクローンまたはダウンロード
2. 必要なPythonパッケージをインストール

```bash
pip install -r requirements.txt
```

## 使い方
1. マイクをPCに接続
2. 下記コマンドでプログラムを実行

```bash
python speechtotext.py
```

3. Qキーを押すと録音・認識を終了します
4. 認識結果は`output.txt`に保存されます

## 設定
- モデルサイズや言語は`speechtotext.py`内の`WhisperTranscriber`初期化時に変更可能です。

## 依存パッケージ
- faster-whisper
- pyaudio
- numpy

## 注意事項
- Whisperモデルのダウンロードには初回のみ時間がかかります。
- マイクデバイスが複数ある場合、デフォルト以外を使う場合は`pyaudio.open`の引数を調整してください。
- 手動でキャッシュを配置する場合は以下の構成になるように格納してください。
  `C:\Users\<ユーザー名>\.cache\huggingface\hub`
    models--Systran--faster-whisper-base
      ├ blobs : データなし
      ├ refs
      │   └ main : コミットハッシュのみが記載されたテキストファイル
      └ snapshots
          └ <コミットハッシュ>
              ├ config.json
              ├ model.bin
              ├ tokenizer.json
              └ vocabulary.txt

    models--Systran--faster-whisper-large-v3
      ├ blobs : データなし
      ├ refs
      │   └ main : コミットハッシュのみが記載されたテキストファイル
      └ snapshots
          └ <コミットハッシュ>
              ├ config.json
              ├ model.bin
              ├ preprocessor_config.json
              ├ tokenizer.json
              └ vocabulary.txt

## 参考
Virtual Audio Cableを利用して作成。ループバック音声はこちらで取得
https://qiita.com/reriiasu/items/1b4cae7205458cf78ade

---

OpenAI Whisper (https://github.com/openai/whisper) を利用しています。
