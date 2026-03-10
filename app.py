import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    AutoModelForCausalLM
)
import csv
import os
from datetime import datetime

LOG_FILE = "logs.csv"

def save_log(service, input_text, output):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "service", "input", "output"])
        writer.writerow([datetime.now().isoformat(), service, input_text[:200], output[:200]])

@st.cache_resource
def load_models():
    summarizer_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return summarizer_tokenizer, summarizer_model, sentiment_pipeline, chat_tokenizer, chat_model

st.set_page_config(page_title="App IA avec logs", layout="wide")
st.title("📊 Application IA améliorée (avec logs)")

with st.spinner("Chargement des modèles... (premier lancement peut être long)"):
    tokenizer, model, sentiment, chat_tokenizer, chat_model = load_models()
st.sidebar.success("Modèles prêts")

st.sidebar.markdown("""
**Limites actuelles :**
- Le chatbot peut donner des réponses non factuelles.
- Le modèle de sentiment peut encore se tromper, mais gère mieux la négation.
- Les résumés sont courts (max 50 mots).
""")

st.sidebar.header("📋 Derniers logs")
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r", encoding='utf-8') as f:
        lines = f.readlines()[-10:]
        for line in lines:
            st.sidebar.text(line.strip())
else:
    st.sidebar.info("Aucun log pour l'instant.")

# Chatbot
st.header("💬 Chatbot local")
user_input = st.text_input("Posez votre question :", key="chat_input")
if user_input:
    if len(user_input) > 500:
        st.error("Question trop longue (max 500 caractères).")
    else:
        try:
            with st.spinner("Le chatbot réfléchit..."):
                inputs = chat_tokenizer.encode(
                    user_input + chat_tokenizer.eos_token,
                    return_tensors='pt',
                    truncation=True,
                    max_length=128
                )
                reply_ids = chat_model.generate(
                    inputs,
                    max_length=150,
                    pad_token_id=chat_tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2
                )
                response = chat_tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
                if not response.strip():
                    response = "Je n'ai pas compris, pouvez‑vous reformuler ?"
            st.write("**Réponse :**", response)
            save_log("chat", user_input, response)
        except Exception as e:
            st.error(f"Erreur chatbot : {e}")
            save_log("chat", user_input, f"ERREUR: {str(e)}")

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
                label = result[0]['label'].lower()
                score = result[0]['score']
                if label == 'positive':
                    emoji = "😃"
                elif label == 'negative':
                    emoji = "😞"
                else:
                    emoji = "😐"
                output = f"{label.capitalize()} ({score:.2f})"
                st.write(f"{emoji} **{label.capitalize()}** (confiance : {score:.2f})")
                save_log("sentiment", sentiment_text, output)
            except Exception as e:
                st.error(f"Erreur analyse : {e}")
                save_log("sentiment", sentiment_text, f"ERREUR: {str(e)}")
    else:
        st.warning("Veuillez entrer une phrase.")
