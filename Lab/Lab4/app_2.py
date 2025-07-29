import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
import torch
from gtts import gTTS
from pydub import AudioSegment
import os
import tempfile
from IPython.display import Audio as IPyAudio
import evaluate
import nltk
import io

nltk.download('punkt')

# ---------------------------------------------
# PDF Text Extraction
# ---------------------------------------------
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# ---------------------------------------------
# MP3 to WAV Conversion for ASR
# ---------------------------------------------
def convert_mp3_to_wav(mp3_file):
    audio = AudioSegment.from_file(mp3_file, format="mp3")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        audio.export(temp_wav.name, format="wav")
        return temp_wav.name

# ---------------------------------------------
# Speech-to-Text using Whisper (tiny model)
# ---------------------------------------------
def transcribe_audio(audio_path):
    whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    result = whisper_pipe(audio_path)
    return result["text"]

# ---------------------------------------------
# QA Pipeline per Language
# ---------------------------------------------
def get_qa_pipeline(language):
    if language == "Tamil":
        model = "ai4bharat/indic-qa-tamil"
    elif language == "German":
        model = "deepset/gelectra-base-germanquad"
    else:
        model = "google/flan-t5-base"
    return pipeline("question-answering", model=model, tokenizer=model)

# ---------------------------------------------
# Text-to-Speech (TTS)
# ---------------------------------------------
def speak(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts.save(temp_audio.name)
        return temp_audio.name

# ---------------------------------------------
# Evaluation Metrics (BLEU + ROUGE-L)
# ---------------------------------------------
def eval_scores(pred, ref):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    
    pred_tok = nltk.word_tokenize(pred)
    ref_tok = nltk.word_tokenize(ref)
    
    b = bleu.compute(predictions=[pred_tok], references=[[ref_tok]])["bleu"]
    r = rouge.compute(predictions=[pred], references=[ref])["rougeL"]
    return round(b * 100, 2), round(r * 100, 2)

# ---------------------------------------------
# Streamlit App
# ---------------------------------------------
st.set_page_config(page_title="Multilingual Voice QA", layout="wide")
st.title("📘 Multilingual Voice-based Question Answering")

# Upload PDF and MP3
pdf_file = st.file_uploader("Upload your PDF document", type=["pdf"])
audio_file = st.file_uploader("Upload your Question (MP3)", type=["mp3"])
language = st.selectbox("Select Question Language", ["Tamil", "German"])

if st.button("Generate Answer"):
    if not pdf_file or not audio_file:
        st.error("Please upload both PDF and MP3 files.")
    else:
        with st.spinner("Processing..."):
            # Extract context from PDF
            context = extract_text_from_pdf(pdf_file)

            # Transcribe MP3 to text
            wav_path = convert_mp3_to_wav(audio_file)
            question = transcribe_audio(wav_path)
            st.markdown(f"### 🔊 Transcribed Question: `{question}`")

            # QA Model
            qa_pipe = get_qa_pipeline(language)
            output = qa_pipe(question=question, context=context)
            answer = output["answer"]
            st.success(f"📘 **Answer:** {answer}")

            # TTS (Tamil → ta, German → de)
            lang_code = "ta" if language == "Tamil" else "de"
            audio_path = speak(answer, lang_code)
            st.audio(audio_path, format="audio/mp3", start_time=0)

            # Evaluation (Optional Reference)
            ref_answer = st.text_input("Enter Reference Answer (for BLEU/ROUGE):")
            if ref_answer:
                bleu, rouge = eval_scores(answer, ref_answer)
                st.write(f"🔵 **BLEU Score**: {bleu}")
                st.write(f"🔴 **ROUGE-L Score**: {rouge}")
