import joblib
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import random

sample_rate = 44100 
duration = 5 

model = joblib.load('SVC_model_34ssamples.joblib')

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=35)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

data_dir = 'C:/Users/Asus/Desktop/6th_sem_project/test'
expected= []
predicted = []

audios = []

for file_name in os.listdir(data_dir):
    audios.append(file_name)

# print(audios)

count = {}

def voicesamples():
    for i in range(15):
        j = random.randint(0, 31)
        expected.append(audios[j])
        print(f"{audios[j]}\n")
        file_path = os.path.join(data_dir, audios[j])
        if file_path.endswith('.flac' or '.wav'):
            extracted_features = extract_features(file_path)
            extracted_features = extracted_features.reshape(1, -1)
            prediction = model.predict(extracted_features)
            predicted.append(prediction)
            for k in prediction:
                if k not in count:
                    count[k] = 1
                else:
                    count[k] += 1

def record():
    print("Recording...")
    audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
    sd.wait()
    file_path = "predict.flac"
    sf.write(file_path, audio, sample_rate)
    new_features = extract_features("C:/Users/Asus/Desktop/6th_sem_project/predict.flac")
    print(new_features)
    new_features = new_features.reshape(1, -1)
    new_prediction = model.predict(new_features)
    print("Prediction:", new_prediction)

print()
voicesamples()
print("Files\t\tPredicted")
for i in range(len(expected)):
    print(expected[i],"\t",predicted[i])
print("count: ", count)
# record()

# voicesamples()
# print(count)

# new_features2 = extract_features("C:/Users/Asus/Desktop/6th_sem_project/voice samples/Aditya/voice_sample17.wav")
# print("\n",new_features2)

# new_features3 = extract_features("C:/Users/Asus/Desktop/6th_sem_project/voice samples/Aditya/voice_sample18.wav")
# print("\n",new_features3)
# print(count)
