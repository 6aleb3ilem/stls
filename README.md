
# systeme de traduction en langue des signes

Ce projet vise à traduire les gestes de la langue des signes américaine (ASL) en texte. L'application utilise MediaPipe pour la détection des gestes de la main et offre une interface simple pour traduire les gestes en texte.

## Configuration de l'environnement

1. **Créer et activer un environnement Conda :**
    ```bash
    conda create -n mediapipe python=3.10
    conda activate mediapipe
    ```

2. **Installer les paquets requis :**
    ```bash
    pip install -r requirements.txt
    ```


## Utilisation

### Lancer l'Application

#### Pour démarrer l'application, exécutez :
##### 1 .
```bash

         python main.py

```
## Screenshots

![App Screenshot](https://github.com/6aleb3ilem/stls/assets/121716974/6dcf52fb-d014-4e6b-8a7b-61a01a38390c)

![runing with main2](https://github.com/6aleb3ilem/stls/assets/121716974/d564be85-285e-42e1-9bb7-d5eace3911c3)
![runing with main 3](https://github.com/6aleb3ilem/stls/assets/121716974/ae579a27-2ab5-4703-bd81-d4786bd569d3)
``` bash 
        python main.py --help

```
        Options
        -h, --help: Afficher le message d'aide et quitter.
        -t TIMING, --timing TIMING: Définir le seuil de timing.
        -wi WIDTH, --width WIDTH: Définir la largeur de la webcam.
        -he HEIGHT, --height HEIGHT: Définir la hauteur de la webcam.
        -f FPS, --fps FPS: Définir les FPS de la webcam.
        Contrôles
        esc : Quitter l'application.
        R: Commencer à détecter les gestes de la main.
        C: Effacer tous les mots de sortie.
        M: Changer de mode entre les chiffres et les lettres.
        Back space : Supprimer le dernier mot.


#### ou 

 #### 2. Interface Web

    
    Vous pouvez également exécuter l'interface web en utilisant Streamlit :
```bash
        streamlit run app.py
```
 ![running with streamlit 1](https://github.com/6aleb3ilem/stls/assets/121716974/692c9242-a11a-4282-bed2-ba6c2fd00a21)
 ![running with streamlit 2](https://github.com/6aleb3ilem/stls/assets/121716974/590b5e5c-bb65-487d-ad03-725c59305014)

     
