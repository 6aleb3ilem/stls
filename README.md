# Projet de Traduction ASL vers Texte

Ce projet vise à traduire les gestes de la langue des signes américaine (ASL) en texte. L'application utilise MediaPipe pour la détection des gestes de la main et offre une interface simple pour traduire les gestes en texte.

## Installation

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

Pour démarrer l'application, exécutez :
```bash
python main.py


python main.py --help


Options
-h, --help: Afficher le message d'aide et quitter.
-t TIMING, --timing TIMING: Définir le seuil de timing.
-wi WIDTH, --width WIDTH: Définir la largeur de la webcam.
-he HEIGHT, --height HEIGHT: Définir la hauteur de la webcam.
-f FPS, --fps FPS: Définir les FPS de la webcam.
Contrôles
Échap: Quitter l'application.
R: Commencer à détecter les gestes de la main.
C: Effacer tous les mots de sortie.
M: Changer de mode entre les chiffres et les lettres.
Retour arrière: Supprimer le dernier mot.
Structure du Projet
main.py: Le script principal pour exécuter l'application.
requirements.txt: Contient toutes les dépendances requises pour le projet.
