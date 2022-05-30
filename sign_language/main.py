"""
Description: Sign Language Recognition

This work is a combined effort of
Authors:
    Adish Pathare
    Anubhuti Puppalwar
    Navneet Desai
"""


import time
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os
import mediapipe
from sklearn.model_selection import train_test_split


class Constants:
    """
    Class that holds all the constants
    """
    SUMMARIZE = True
    CAMERA = cv2.VideoCapture(0)
    HOLISTIC = mediapipe.solutions.holistic
    DRAW = mediapipe.solutions.drawing_utils
    DATA = os.path.join('DATA')
    PHRASES = np.array(['hello', 'thanks', 'you'])
    NUMBER_OF_PHRASES = 3
    TOTAL_SEQUENCES = 30
    TOTAL_FRAMES = 30
    TOTAL_LANDMARKS = 1662
    DEFAULT_POSE_LANDMARKS = 33
    DEFAULT_HAND_LANDMARKS = 21
    DEFAULT_FACE_LANDMARKS = 468
    EPOCHS = 300


def detect(img, model):
    """
    Converts an img from BGR to RGB format
    so that the model could process it.
    Returns the results
    :param img: img
    :param model: model
    :return: original img and the results of processing
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return model.process(img)


def show_contours(img, result):
    """
    This function draws the landmarks on the face, posture and the hands
    :param img:
    :param result:
    :return:
    """
    Constants.DRAW.draw_landmarks(img, result.left_hand_landmarks, Constants.HOLISTIC.HAND_CONNECTIONS)
    Constants.DRAW.draw_landmarks(img, result.right_hand_landmarks, Constants.HOLISTIC.HAND_CONNECTIONS)
    Constants.DRAW.draw_landmarks(img, result.face_landmarks, Constants.HOLISTIC.FACEMESH_CONTOURS)
    Constants.DRAW.draw_landmarks(img, result.pose_landmarks, Constants.HOLISTIC.POSE_CONNECTIONS)


def return_flattened_keypoints(landmarks):
    """
    This function extracts landmarks from facial, hand and postural landmarks and returns
    an numpy concatenation of flattened landmarks
    :param landmarks:
    :return:
    """
    left_hand = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.left_hand_landmarks.landmark]).flatten() \
        if landmarks.left_hand_landmarks else np.zeros(Constants.DEFAULT_HAND_LANDMARKS * 3)
    right_hand = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.right_hand_landmarks.landmark]).flatten() \
        if landmarks.right_hand_landmarks else np.zeros(Constants.DEFAULT_HAND_LANDMARKS * 3)
    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks.pose_landmarks.landmark]).flatten() \
        if landmarks.pose_landmarks else np.zeros(Constants.DEFAULT_POSE_LANDMARKS * 4)
    face = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.face_landmarks.landmark]).flatten() \
        if landmarks.face_landmarks else np.zeros(Constants.DEFAULT_FACE_LANDMARKS * 3)
    # print([pose, face, left_hand, right_hand])
    return np.concatenate([pose, face, left_hand, right_hand])


def create_directories():
    """
    Helper function for recursively creating directories
    :return: None
    """
    for phrase in Constants.PHRASES:
        for sequence in range(Constants.TOTAL_SEQUENCES):
            os.makedirs(f'{Constants.DATA}/{phrase}/{str(sequence)}')


def _capture_videos(camera):
    """
    This is the first method that should be called before processing the model.
    For each phrase in the knowledge base, it creates Constants.TOTAL_SEQUENCES number
    of sequences(videos) and Constants.TOTAL_FRAMES TOTAL_FRAMES for each sequences.
    The method sleeps for about 3 seconds before starting the iterations for each
    phrase.
    The method reads a frame from the CAMERA instance. draws landmarks on the screen.
    The keypoints are processed, flattened and stored in .npy files for each sequence.
    The user can quit at any time by pressing 'q'

    Exit: The method releases CAMERA and destroys all opencv windows
    :param camera:
    :return:
    """
    with Constants.HOLISTIC.Holistic(min_detection_confidence=0.35, min_tracking_confidence=0.35) as holistic:
        for phrase in Constants.PHRASES:
            print(f'Starting for : {phrase}')
            time.sleep(3)
            for sequence in range(Constants.TOTAL_SEQUENCES):
                for index in range(Constants.TOTAL_FRAMES):
                    _, frame = camera.read()
                    result = detect(frame, holistic)
                    cv2.imshow('LIVE: ', frame)
                    show_contours(frame, result)
                    print(f'Collecting frames for {phrase}, sequence: {sequence} index: {index}')

                    keypoints = return_flattened_keypoints(result)

                    npy_path = os.path.join(Constants.DATA, phrase, str(sequence), str(index))
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        break
        cv2.destroyAllWindows()
        camera.release()
        return result


def test_model(camera, model):
    """
    This functions tests the model
    :param camera:
    :param model:
    :return:
    """
    sequence_storage = []

    with Constants.HOLISTIC.Holistic(min_detection_confidence=0.35, min_tracking_confidence=0.35) as holistic:
        while camera.isOpened():
            _, frame = camera.read()
            result = detect(frame, holistic)

            show_contours(frame, result)
            keypoints = return_flattened_keypoints(result)

            sequence_storage.append(keypoints)
            sequence_storage = sequence_storage[-Constants.TOTAL_FRAMES:]

            if len(sequence_storage) == Constants.TOTAL_FRAMES:
                result = model.predict(np.expand_dims(sequence_storage, axis=0))[0]
                # if the result is over a threshold, print the output
                if result[np.argmax(result)] > 0.75:
                    print(f'Detecting phrase: {Constants.PHRASES[np.argmax(result)]}')
            cv2.imshow('LIVE: ', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        camera.release()


def main():
    """
    Main method: Driver function
    :return:
    """
    # following line should be uncommented for capturing videos
    # capture_videos()

    sequences, phrases = extract_sequences_and_labels()
    x = np.array(sequences)
    y = to_categorical(phrases).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    callback = TensorBoard(log_dir=os.path.join('tensorboard_data' + f'{time.time()}'))
    model = run_model(x_test, x_train, callback, y_train)
    test_model(Constants.CAMERA, model)


def run_model(x_test, x_train, callback, y_train):
    """
    Builds a sequential and adds layers.
    """
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='elu',
                   input_shape=(Constants.TOTAL_FRAMES, Constants.TOTAL_LANDMARKS)))
    model.add(LSTM(64, return_sequences=True, activation='elu'))
    model.add(LSTM(128, return_sequences=False, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(128, activation='elu'))
    model.add(Dense(Constants.NUMBER_OF_PHRASES, activation='softmax'))
    model.compile(optimizer='Adadelta', metrics=['categorical_accuracy'], loss='categorical_crossentropy')
    model.fit(x_train, y_train, epochs=Constants.EPOCHS, callbacks=[callback])
    if Constants.SUMMARIZE:
        model.summary()
    model.predict(x_test)
    model.save('sign_language.h5')
    return model


def extract_sequences_and_labels():
    """
    This method loads TOTAL_FRAMES for each of the PHRASES in Constants.PHRASES
    :return:
    """
    sequences, phrases = [], []
    for phrase in Constants.PHRASES:
        for sequence in range(Constants.TOTAL_SEQUENCES):
            frames = []
            for index in range(Constants.TOTAL_FRAMES):
                frames.append(np.load(os.path.join(Constants.DATA, phrase, str(sequence), "{}.npy".format(index))))
            # append the TOTAL_FRAMES and the corresponding phrase
            sequences.append(frames)
            phrases.append(Constants.PHRASES.tolist().index(phrase))
    return sequences, phrases


def capture_videos():
    """
    Captures the videos for training the model
    Should be commented from the main function after data has already been recorded
    :return: None
    """
    results = _capture_videos(Constants.CAMERA)
    _ = return_flattened_keypoints(results)
    create_directories()


if __name__ == '__main__':
    main()
