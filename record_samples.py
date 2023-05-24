import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np

sample_rate = 44100 
duration = 10 

for i in range(10):
	print("Recording...")
	audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
	sd.wait()
	file_path = f"C:/Users/Asus/Desktop/6th_sem_project/audio/voice_sample{i}.flac"
	sf.write(file_path, audio, sample_rate)
	print(f"Voice sample saved as {file_path}")
