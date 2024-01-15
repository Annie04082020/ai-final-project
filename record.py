import pyaudio
import wave
import requests
import json
# from main import predict

def record_audio(file_path, duration=5):
    """錄製音檔"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    print("錄音中...")
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("錄音完成.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # 將音檔寫入 wav 檔案
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == "__main__":
    # 定義錄音的音檔路徑
    audio_file_path = "recorded/recorded_audio.wav"
    # 錄音並保存音檔
    record_audio(audio_file_path)