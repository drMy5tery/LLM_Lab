# app.py
import streamlit as st
import pymupdf as fitz
from io import BytesIO
from gtts import gTTS
import evaluate, json
from datetime import datetime
from faster_whisper import WhisperModel
from transformers import pipeline

# ────────── ASR SETUP ──────────
@st.cache_resource
def init_asr():
    return WhisperModel(model_size_or_path="small", device="cpu", compute_type="int8")
asr = init_asr()

# ────────── QA MODELS ──────────
@st.cache_resource
def load_qa(model_id):
    return pipeline("question-answering", model=model_id, device=0)  # GPU if available

qa_models = {
    "English – DistilBERT‑SQuAD": load_qa("distilbert-base-cased-distilled-squad"),
    "Spanish – DistilBETO‑SQuAD": load_qa("mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"),
    "Kannada – Squad‑BERT": load_qa("l3cube-pune/kannada-question-answering-squad-bert")
}

LANG_CODES = {
    "English – DistilBERT‑SQuAD": "en",
    "Spanish – DistilBETO‑SQuAD": "es",
    "Kannada – Squad‑BERT": "kn"
}

QNA_HISTORY = []

# ────────── UTILITIES ──────────
def extract_text(pdf):
    data = pdf.read()
    doc = fitz.open(stream=data, filetype="pdf")
    return " ".join(page.get_text() for page in doc)

def transcribe(audio_bytes):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        fname = tmp.name
    segs, _ = asr.transcribe(fname)
    return " ".join(seg.text for seg in segs).strip()

def text_to_speech(answer, lang):
    buf = BytesIO()
    try:
        tts = gTTS(text=answer, lang=lang)
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"TTS failed for lang '{lang}': {e}")
        return None

def compute_metrics(answer, ref):
    bleu = evaluate.load("bleu").compute(predictions=[answer], references=[ref])["bleu"]
    rouge = evaluate.load("rouge").compute(predictions=[answer], references=[ref])["rougeL"]
    return bleu, rouge

# ────────── STREAMLIT UI ──────────
st.set_page_config("Local Voice-QA Multilingual", layout="centered")
st.title("🗣️ Voice‑Based PDF QA — English / Spanish / Kannada")

model_name = st.selectbox("Choose Model/Language", list(qa_models.keys()))
qa = qa_models[model_name]
tts_lang = LANG_CODES[model_name]

pdf = st.file_uploader("Upload PDF", type="pdf")
if pdf:
    context = extract_text(pdf)
    st.success("📄 PDF text loaded.")

    audio_in = st.audio_input("Hold to ask a question")
    if audio_in:
        audio_bytes = audio_in.read()
        with st.spinner("Transcribing..."):
            question = transcribe(audio_bytes)
        st.write("🗣 Raw transcription:", repr(question))

        if not question:
            st.error("⚠️ No speech captured. Please try again.")
        else:
            st.markdown(f"**You asked:** {question}")
            with st.spinner("Answering..."):
                result = qa(question=question, context=context)
            answer = result["answer"]
            st.markdown(f"**🧠 Answer:** {answer}")

            buf = text_to_speech(answer, tts_lang)
            if buf:
                st.audio(buf, format="audio/mp3")

            ref = st.text_input("Reference answer (optional)", key="ref")
            if ref:
                b, r = compute_metrics(answer, ref)
                st.write(f"📊 BLEU: {b:.3f} • ROUGE‑L: {r:.3f}")

            QNA_HISTORY.append({
                "time": datetime.now().isoformat(),
                "model": model_name,
                "question": question,
                "answer": answer
            })

if QNA_HISTORY:
    st.sidebar.header("📜 Q&A History")
    st.sidebar.download_button("Export Log", json.dumps(QNA_HISTORY, indent=2), "qna_log.json")
