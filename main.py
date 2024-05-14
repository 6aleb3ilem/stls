import sys
import cv2
import argparse
import numpy as np
import mediapipe as mp

from utils import load_model
from utils import calc_landmark_list, draw_landmarks, draw_info_text

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Couleurs au format RGB
NOIR = (0, 0, 0)
ROUGE = (255, 0, 0)
VERT = (0, 255, 0)
BLEU = (0, 0, 255)
JAUNE = (0, 255, 255)
BLANC = (255, 255, 255)

# Constantes
POLICE = cv2.FONT_HERSHEY_SIMPLEX
MAX_MAINS = 1
confiance_detection_min = 0.6
confiance_suivi_min = 0.5

MODEL_PATH = "./classifier"
model_letter_path = f"{MODEL_PATH}/classify_letter_model.p"
model_number_path = f"{MODEL_PATH}/classify_number_model.p"

# Personnalisez votre entrée
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--timing', type=int, default=8, help='Seuil de synchronisation')
    parser.add_argument('-wi', '--width', type=int, default=800, help='Largeur de la webcam')
    parser.add_argument('-he', '--height', type=int, default=600, help='Hauteur de la webcam')
    parser.add_argument('-f', '--fps', type=int, default=30, help='FPS de la webcam')
    opt = parser.parse_args()
    return opt

def get_output(idx):
    global _output, output, TIMING
    key = []
    for i in range(len(_output[idx])):
        character = _output[idx][i]
        counts = _output[idx].count(character)

        # Ajouter un caractère à la clé s'il dépasse le 'SEUIL DE SYNCHRONISATION'
        if (character not in key) or (character != key[-1]):
            if counts > TIMING:
                key.append(character)

    # Ajouter le caractère clé au texte de sortie
    text = ""
    for character in key:
        if character == "?":
            continue
        text += str(character).lower()

    # Ajouter le mot à la liste de sortie
    if text != "":
        _output[idx] = []
        output.append(text.title())
    return None

def recognize_gesture(image, results, numberMode, model_letter_path, model_number_path):
    global mp_drawing, current_hand
    global output, _output

    multi_hand_landmarks = results.multi_hand_landmarks
    multi_handedness = results.multi_handedness

    # Charger le modèle de classification
    letter_model = load_model(model_letter_path)
    number_model = load_model(model_number_path)

    _gesture = []
    data_aux = []

    # Nombre de mains
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

            # Obtenir les coordonnées (x, y) des points de repère de la main
            x_values = [lm.x for lm in current_select_hand.landmark]
            y_values = [lm.y for lm in current_select_hand.landmark]

            # Obtenir les valeurs minimales et maximales
            min_x = int(min(x_values) * w)
            max_x = int(max(x_values) * w)
            min_y = int(min(y_values) * h)
            max_y = int(max(y_values) * h)

            # Dessiner les informations textuelles
            cv2.putText(image, f"Hand No: #{idx}", (min_x - 10, max_y + 30), POLICE, 0.5, VERT, 2)
            cv2.putText(image, f"{handness} Hand", (min_x - 10, max_y + 60), POLICE, 0.5, VERT, 2)

            # Retourner la main gauche en main droite
            if handness == 'Left':
                x_values = list(map(lambda x: 1 - x, x_values))
                min_x -= 10

            # Créer une augmentation des données pour la main corrigée
            for i in range(len(current_select_hand.landmark)):
                data_aux.append(x_values[i] - min(x_values))
                data_aux.append(y_values[i] - min(y_values))

            if not numberMode:
                # Prédiction des alphabets
                prediction = letter_model.predict([np.asarray(data_aux)])
                gesture = str(prediction[0]).title()
                gesture = gesture if gesture != 'Unknown_Letter' else '?'
            else:
                # Prédiction des nombres
                prediction = number_model.predict([np.asarray(data_aux)])
                gesture = str(prediction[0]).title()
                gesture = gesture if gesture != 'Unknown_Number' else '?'

            # Dessiner le cadre de délimitation
            cv2.rectangle(image, (min_x - 20, min_y - 10), (max_x + 20, max_y + 10), NOIR, 4)
            image = draw_info_text(image, [min_x - 20, min_y - 10, max_x + 20, max_y + 10], gesture)

            _gesture.append(gesture)

    # Le nombre de mains diminue, créer un "ESPACE"
    if isDecreased == True:
        if current_hand == 1:
            get_output(0)

    # Le nombre de mains est le même, ajouter le geste
    else:
        if results.multi_hand_landmarks is not None:
            _output[0].append(_gesture[0])

    # Suivre le nombre de mains
    if results.multi_hand_landmarks:
        current_hand = len(multi_hand_landmarks)
    else:
        current_hand = 0

    return image

if __name__ == '__main__':
    opt = parse_opt()

    global TIMING
    TIMING = opt.timing
    print(f"Le seuil de synchronisation est de {TIMING} images.")

    # Arguments de la webcam
    fps = opt.fps
    webcam_width = opt.width
    webcam_height = opt.height

    _output = [[], []]
    output = []
    quitApp = False

    frame_array = []
    current_hand = 0
    numberMode = False

    # Entrée de la webcam
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    capture.set(cv2.CAP_PROP_FPS, fps)

    # Appuyez sur 'r' si vous êtes prêt
    while True:
        success, frame = capture.read()
        frame = cv2.flip(frame, 1)

        # Configurer le texte
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'Pret? Appuyez sur "R"'

        # Obtenir les limites de ce texte
        textsize = cv2.getTextSize(text, font, 1.3, 3)[0]

        # Obtenir les coordonnées en fonction des limites
        textX = (frame.shape[1] - textsize[0]) // 2
        textY = (frame.shape[0] + textsize[1]) // 2

        cv2.putText(
            frame, text, (textX, textY),
            font, 1.3, VERT, 3,
            cv2.LINE_AA
        )
        cv2.imshow('Reconnaissance de gestes:', frame)

        # Entrée utilisateur depuis le clavier
        key = cv2.waitKey(5) & 0xFF
        if key == ord('r'):
            break

        # Appuyez sur 'Échap' pour quitter
        if key == 27:
            quitApp = True
            break

    # Supprimer la fenêtre de démarrage
    cv2.destroyAllWindows()
    if quitApp == True:
        capture.release()
        quit()

    with mp_hands.Hands(
        min_detection_confidence=confiance_detection_min,
        min_tracking_confidence=confiance_suivi_min, 
        max_num_hands=MAX_MAINS
    ) as hands:
        while capture.isOpened():
            success, image = capture.read()
            if not success:
                print("Ignorer le cadre de la caméra vide.")
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
                print(f"{error}, ligne {exc_tb.tb_lineno}")

            # Dessiner la sortie
            output_text = str(output)
            output_size = cv2.getTextSize(output_text, POLICE, 0.5, 2)[0]
            cv2.rectangle(image, (5, 0), (10 + output_size[0], 10 + output_size[1]), JAUNE, -1)
            cv2.putText(image, output_text, (10, 15), POLICE, 0.5, NOIR, 2)

            # Dessiner le mode actuel
            mode_text = f"Mode nombre : {numberMode}"
            mode_size = cv2.getTextSize(mode_text, POLICE, 0.5, 2)[0]
            cv2.rectangle(image, (5, 45), (10 + mode_size[0], 10 + mode_size[1]), JAUNE, -1)
            cv2.putText(image, mode_text, (10, 40), POLICE, 0.5, NOIR, 2)

            # Afficher le cadre
            cv2.imshow('Systeme de traduction en language des signes ', image)
            key = cv2.waitKey(5) & 0xFF

            # Appuyez sur 'Échap' pour quitter
            if key == 27:
                break

            # Appuyez sur 'Retour arrière' pour supprimer le dernier caractère
            if key == 8:
                output.pop()

            # Appuyez sur 'm' pour changer de mode
            if key == ord('m'):
                numberMode = not numberMode

            # Appuyez sur 'c' pour effacer la sortie
            if key == ord('c'):
                output.clear()

    cv2.destroyAllWindows()
    capture.release()

    # Imprimer la sortie
    print(f"Reconnaissance de gestes:\n{' '.join(output)}")
