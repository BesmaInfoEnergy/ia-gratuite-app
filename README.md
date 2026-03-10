# IA Gratuite App

Application web interactive proposant trois outils d'intelligence artificielle gratuits et exécutables localement ou dans le cloud :

- **Chatbot** conversationnel (modèle GPT4All)
- **Résumé automatique** de textes longs (modèle T5-small)
- **Analyse de sentiment** (modèle DistilBERT)

Tous les modèles fonctionnent en local sans connexion internet après le premier téléchargement. L'application enregistre chaque requête dans un fichier `logs.csv` pour suivi.

## Fonctionnalités

- Interface simple et intuitive avec [Streamlit](https://streamlit.io)
- Chatbot capable de répondre à des questions en langage naturel
- Résumé de textes jusqu'à 2000 caractères
- Détection de sentiment positif/négatif avec score de confiance
- Journalisation automatique des interactions (horodatage, service, entrée, sortie)

## Installation en local

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/votre-nom/ia-gratuite-app.git
   cd ia-gratuite-app
   Créer un environnement virtuel (recommandé)
   ```

bash
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
Installer les dépendances

bash
pip install -r requirements.txt
Lancer l'application

bash
streamlit run app.py
L'application s'ouvrira automatiquement dans votre navigateur à l'adresse http://localhost:8501.

Utilisation
Chatbot : saisissez une question dans le champ prévu, appuyez sur Entrée, la réponse apparaît.

Résumé : collez un texte long (max 2000 caractères) et cliquez sur "Résumer".

Analyse de sentiment : entrez une phrase et cliquez sur "Analyser le sentiment".

Les logs sont visibles dans la barre latérale (dernières 10 lignes) et stockés dans le fichier logs.csv.

Déploiement sur Streamlit Cloud (gratuit)
Poussez le code sur un dépôt GitHub public (déjà fait si vous lisez ceci).

Rendez-vous sur share.streamlit.io.

Connectez-vous avec votre compte GitHub, sélectionnez ce dépôt, la branche main et le fichier app.py.

Cliquez sur Deploy.

Une fois déployé, vous obtiendrez une URL publique (ex. https://ia-gratuite-app.streamlit.app).

Note : Le modèle GPT4All peut rencontrer des problèmes sur les serveurs cloud (dépendances système manquantes). Si le chatbot ne fonctionne pas en ligne, remplacez-le par un modèle purement Python comme DialoGPT-small (voir cette section dans la documentation complète).

Technologies utilisées
Streamlit – Interface web

Transformers (Hugging Face) – Modèles de NLP

GPT4All – Chatbot local

FastAPI (utilisé dans les étapes intermédiaires, non requis ici)

Python 3.10+

Auteur
Créé par [BesmaInfoEnergy] dans le cadre d'une roadmap d'apprentissage de 7 jours sur les IA génératives.

Licence
Ce projet est librement réutilisable à des fins éducatives. Les modèles pré-entraînés appartiennent à leurs auteurs respectifs (licences MIT, Apache 2.0, etc.).
