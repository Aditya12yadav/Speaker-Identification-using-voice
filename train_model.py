import librosa
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os
import joblib

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=35)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

data_dir = 'C:/Users/Asus/Desktop/6th_sem_project/voice samples'

features = []
labels = []

for user_folder in os.listdir(data_dir):
    user_folder_path = os.path.join(data_dir, user_folder)
    if os.path.isdir(user_folder_path):
        for file_name in os.listdir(user_folder_path):
            file_path = os.path.join(user_folder_path, file_name)
            if file_path.endswith('.flac' or '.wav'):
                extracted_features = extract_features(file_path)
                features.append(extracted_features)
                labels.append(user_folder)
                print(f"Done extracting features from {file_name}")

# print("Done")
X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

joblib.dump(model, 'SVC_model_34ssamples.joblib')

y_train_pred = model.predict(X_train)
train_accuracy = np.mean(y_train_pred == y_train)

y_test_pred = model.predict(X_test)
test_accuracy = np.mean(y_test_pred == y_test)

print("\nTraining Accuracy:", train_accuracy)
print("\nTesting Accuracy:", test_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# print("\n\n")

# new_features = extract_features("C:/Users/Asus/Desktop/6th_sem_project/voice samples/4/4-14.flac")
# new_features = new_features.reshape(1, -1)
# new_prediction = model.predict(new_features)
# print("Prediction:", new_prediction)
