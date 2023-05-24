# Speaker-Identification-using-voice

This code performs audio classification using the MFCC (Mel-frequency cepstral coefficients) features and a Support Vector Machine (SVM) classifier. Here is a brief description of the code:

1. The code defines a function `extract_features(file_path)` that takes an audio file path as input and extracts the MFCC features using the Librosa library.
2. The data directory is specified, where the voice samples are stored.
3. Features and labels lists are initialized to store the extracted features and corresponding labels.
4. The code iterates over the user folders in the data directory, then for each audio file in each user folder, it extracts the features using the `extract_features()` function and appends them to the features list, along with the corresponding label.
5. The extracted features are converted to NumPy arrays for further processing.
6. The features and labels are split into training and testing sets using the `train_test_split()` function from scikit-learn.
7. An SVM classifier with a linear kernel is initialized.
8. The model is trained on the training data using the `fit()` function.
9. The trained model is saved to a file using the `joblib.dump()` function.
10. Predictions are made on the training and testing data, and the accuracy is calculated by comparing the predicted labels with the actual labels.
11. The training and testing accuracies are printed.
12. The classification report, including precision, recall, F1-score, and support, is printed using the `classification_report()` function from scikit-learn.

Overall, this code demonstrates the process of extracting MFCC features from audio files and training an SVM classifier for audio classification. It uses Librosa for audio processing and scikit-learn for data splitting, model training, and evaluation. The SVM model can then be used to predict the labels for new audio samples.

The additional files are:
record.py: to record 10 voice samples of a user and save it which can be further moved to the samples folder to be used for training
predict.py: to predict the speaker using an already existing voice sample or record a voice sample and predict the speaker
