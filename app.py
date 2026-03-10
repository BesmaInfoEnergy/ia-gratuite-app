# Lancez : streamlit run app_logs.py
# Vérifiez que le fichier logs.csv est créé et s’enrichit à chaque utilisation.

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from gpt4all import GPT4All
import csv
import os
from datetime import datetime

# Fichier de logs
LOG_FILE = "logs.csv"

def save_log(service, input_text, output):
    """Enregistre une requête dans le CSV."""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "service", "input", "output"])
        writer.writerow([datetime.now().isoformat(), service, input_text[:200], output[:200]])  # tronquer pour lisibilité

@st.cache_resource
def load_models():
    summarizer_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    chat_model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
    return summarizer_tokenizer, summarizer_model, sentiment_pipeline, chat_model

st.set_page_config(page_title="App IA avec logs", layout="wide")
st.title("📊 Application IA améliorée (avec logs)")

# Chargement des modèles
with st.spinner("Chargement des modèles..."):
    tokenizer, model, sentiment, chat = load_models()
st.sidebar.success("Modèles prêts")

# Affichage des logs récents dans la sidebar
st.sidebar.header("📋 Derniers logs")
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r", encoding='utf-8') as f:
        lines = f.readlines()[-10:]  # 10 dernières lignes
        for line in lines:
            st.sidebar.text(line.strip())
else:
    st.sidebar.info("Aucun log pour l'instant.")

# --------------------------------------------------
# Chatbot
st.header("💬 Chatbot local")
user_input = st.text_input("Posez votre question :", key="chat_input")
if user_input:
    if len(user_input) > 500:
        st.error("Question trop longue (max 500 caractères).")
    else:
        try:
            with st.spinner("Le chatbot réfléchit..."):
                response = chat.generate(user_input, max_tokens=200)  # Limiter la longueur
            st.write("**Réponse :**", response)
            save_log("chat", user_input, response)
        except Exception as e:
            st.error(f"Erreur chatbot : {e}")
            save_log("chat", user_input, f"ERREUR: {str(e)}")

# --------------------------------------------------
# Résumé
st.header("📝 Résumé de texte")
long_text = st.text_area("Collez le texte à résumer (max 2000 caractères) :", height=150)
if st.button("Résumer"):
    if long_text:
        if len(long_text) > 2000:
            st.error("Texte trop long, veuillez réduire à 2000 caractères.")
        else:
            try:
                with st.spinner("Génération du résumé..."):
                    input_text = "summarize: " + long_text
                    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
                    summary_ids = model.generate(
                        inputs,
                        max_length=50,
                        min_length=20,
                        length_penalty=2.0,
                        num_beams=4,
                        early_stopping=True
                    )
                    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                st.write("**Résumé :**", summary)
                save_log("summarizer", long_text[:100] + "...", summary)
            except Exception as e:
                st.error(f"Erreur résumé : {e}")
                save_log("summarizer", long_text[:100] + "...", f"ERREUR: {str(e)}")
    else:
        st.warning("Veuillez entrer un texte.")

# --------------------------------------------------
# Sentiment
st.header("😊 Analyse de sentiment")
sentiment_text = st.text_input("Entrez une phrase :", key="sentiment_input")
if st.button("Analyser le sentiment"):
    if sentiment_text:
        if len(sentiment_text) > 500:
            st.error("Phrase trop longue (max 500 caractères).")
        else:
            try:
                with st.spinner("Analyse en cours..."):
                    result = sentiment(sentiment_text)
                label = result[0]['label']
                score = result[0]['score']
                emoji = "😃" if label == "POSITIVE" else "😞"
                output = f"{label} ({score:.2f})"
                st.write(f"{emoji} **{label}** (confiance : {score:.2f})")
                save_log("sentiment", sentiment_text, output)
            except Exception as e:
                st.error(f"Erreur analyse : {e}")
                save_log("sentiment", sentiment_text, f"ERREUR: {str(e)}")
    else:
        st.warning("Veuillez entrer une phrase.")
