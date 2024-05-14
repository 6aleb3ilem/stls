import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from utils import load_model, calc_landmark_list, draw_landmarks, draw_info_text

# Initialize Mediapipe and models
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Load models
MODEL_PATH = "./classifier"
model_letter_path = f"{MODEL_PATH}/classify_letter_model.p"
model_number_path = f"{MODEL_PATH}/classify_number_model.p"
letter_model = load_model(model_letter_path)
number_model = load_model(model_number_path)

# Constants
NOIR = (0, 0, 0)
ROUGE = (255, 0, 0)
VERT = (0, 255, 0)
BLEU = (0, 0, 255)
JAUNE = (0, 255, 255)
BLANC = (255, 255, 255)
POLICE = cv2.FONT_HERSHEY_COMPLEX

# Global variables
output = []
_output = [[]]
current_hand = 0
numberMode = False
quitApp = False

# Functions
def get_output(idx, TIMING):
    global _output, output
    key = []
    for i in range(len(_output[idx])):
        character = _output[idx][i]
        counts = _output[idx].count(character)
        if (character not in key) or (character != key[-1]):
            if counts > TIMING:
                key.append(character)
    text = ""
    for character in key:
        if character == "?":
            continue
        text += str(character).lower()
    if text != "":
        _output[idx] = []
        output.append(text.title())
    return None

def recognize_gesture(image, results, numberMode, model_letter_path, model_number_path):
    global mp_drawing, current_hand
    global output, _output

    multi_hand_landmarks = results.multi_hand_landmarks
    multi_handedness = results.multi_handedness

    letter_model = load_model(model_letter_path)
    number_model = load_model(model_number_path)

    _gesture = []
    data_aux = []

    isIncreased = False
    isDecreased = False

    if current_hand != 0:
        if results.multi_hand_landmarks is None:
            isDecreased = True
        else:
            if len(multi_hand_landmarks) > current_hand:
                isIncreased = True
            elif len(multi_hand_landmarks) < current_hand:
                isDecreased = True

    if results.multi_hand_landmarks:
        h, w, _ = image.shape
        for idx in reversed(range(len(multi_hand_landmarks))):
            current_select_hand = multi_hand_landmarks[idx]
            handness = multi_handedness[idx].classification[0].label

            landmark_list = calc_landmark_list(image, current_select_hand)
            image = draw_landmarks(image, landmark_list)

            x_values = [lm.x for lm in current_select_hand.landmark]
            y_values = [lm.y for lm in current_select_hand.landmark]

            min_x = int(min(x_values) * w)
            max_x = int(max(x_values) * w)
            min_y = int(min(y_values) * h)
            max_y = int(max(y_values) * h)

            cv2.putText(image, f"Main numero: #{idx}", (min_x - 10, max_y + 30), POLICE, 0.8, VERT, 2)
            cv2.putText(image, f"{handness} main", (min_x - 10, max_y + 60), POLICE, 0.8, VERT, 2)

            if handness == 'Left':
                x_values = list(map(lambda x: 1 - x, x_values))
                min_x -= 10

            for i in range(len(current_select_hand.landmark)):
                data_aux.append(x_values[i] - min(x_values))
                data_aux.append(y_values[i] - min(y_values))

            if not numberMode:
                prediction = letter_model.predict([np.asarray(data_aux)])
                gesture = str(prediction[0]).title()
                gesture = gesture if gesture != 'Unknown_Letter' else '?'
            else:
                prediction = number_model.predict([np.asarray(data_aux)])
                gesture = str(prediction[0]).title()
                gesture = gesture if gesture != 'Unknown_Number' else '?'

            cv2.rectangle(image, (min_x - 20, min_y - 10), (max_x + 20, max_y + 10), NOIR, 4)
            image = draw_info_text(image, [min_x - 20, min_y - 10, max_x + 20, max_y + 10], gesture)

            _gesture.append(gesture)

    if isDecreased:
        if current_hand == 1:
            get_output(0, TIMING)

    else:
        if results.multi_hand_landmarks is not None:
            _output[0].append(_gesture[0])

    if results.multi_hand_landmarks:
        current_hand = len(multi_hand_landmarks)
    else:
        current_hand = 0

    return image

# Streamlit UI
st.title("Reconnaissance des gestes de la main")

# Configuration de la webcam
webcam_width = st.slider('Largeur de la webcam', 400, 1280, 800)
webcam_height = st.slider('Hauteur de la webcam', 300, 720, 600)
fps = st.slider('FPS de la webcam', 5, 60, 30)
TIMING = st.slider('Seuil de synchronisation', 1, 20, 8)

# Boutons de contrôle en dehors de la boucle
start_detection = st.button("Démarrer la détection des gestes de la main", key="start_detection")
quit_button = st.button("Quitter", key="quit")
clear_button = st.button("Effacer", key="clear")
change_mode_button = st.button("Changer de mode", key="change_mode")
delete_last_button = st.button("Supprimer le dernier mot", key="delete_last")

if start_detection:
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    capture.set(cv2.CAP_PROP_FPS, fps)

    frame_placeholder = st.empty()

    with mp_hands.Hands(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        max_num_hands=1
    ) as hands:
        while capture.isOpened():
            success, image = capture.read()
            if not success:
                st.write("Ignorer le cadre de la caméra vide.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                image = recognize_gesture(image, results, numberMode, model_letter_path, model_number_path)
            except Exception as error:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                st.write(f"{error}, ligne {exc_tb.tb_lineno}")

            # Dessiner la sortie
            output_text = ' '.join(output)
            cv2.putText(image, output_text, (10, 40), POLICE, 1.5, BLEU, 3)  # Change text color to blue

            # Dessiner le mode actuel
            mode_text = f"Mode nombre : {numberMode}"
            cv2.putText(image, mode_text, (10, 80), POLICE, 1, BLEU, 2)  # Change text color to blue

            frame_placeholder.image(image, channels="BGR", use_column_width=True)

            if quit_button:
                quitApp = True
                break

            if clear_button:
                output.clear()

            if change_mode_button:
                numberMode = not numberMode

            if delete_last_button:
                if output:
                    output.pop()

    capture.release()

st.write(f"Reconnaissance de gestes:\n{' '.join(output)}")
