import streamlit as st
from utils.pdf_utils import extract_text_from_pdf
from utils.speech_utils import transcribe_audio
from utils.tts_utils import speak_text
from utils.evaluation_utils import evaluate_generated_answers
from models.flan_qa import answer_question_flan
from models.tamil_qa import answer_question_tamil
from models.camembert_qa import answer_question_french
import tempfile

st.set_page_config(page_title="Voice QA System", layout="centered")
st.title("Voice-Based Document QA System")

lang_choice = st.selectbox("Choose Language", ["English", "Tamil", "French"])
uploaded_pdf = st.file_uploader("Upload a PDF Document", type="pdf")
uploaded_audio = st.file_uploader("Upload a Voice Question (WAV)", type=["wav"])
reference_answer = st.text_input("Enter the Reference Answer (for evaluation)")

if uploaded_pdf and uploaded_audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_pdf.read())
        doc_text = extract_text_from_pdf(temp_pdf.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_audio.read())
        question = transcribe_audio(temp_audio.name, lang_choice)

    st.markdown(f"**Recognized Question:** {question}")

    if lang_choice == "English":
        answer = answer_question_flan(doc_text, question)
        speak_text(answer, lang="en")
    elif lang_choice == "Tamil":
        answer = answer_question_tamil(doc_text, question)
        speak_text(answer, lang="ta")
    elif lang_choice == "French":
        answer = answer_question_french(doc_text, question)
        speak_text(answer, lang="fr")

    if answer.strip() == "":
        st.error("No answer was generated. Please try a simpler question or verify the PDF content.")
    else:
        st.success("Answer: " + answer)

        if reference_answer.strip():
            eval_data = [{
                "question": question,
                "reference": reference_answer,
                "generated": answer
            }]
            results = evaluate_generated_answers(lang_choice, eval_data)
            st.markdown("**Evaluation Metrics:**")
            for result in results:
                st.write(f"**Question:** {result['question']}")
                st.write(f"**BLEU Score:** {result['bleu']:.4f}")
                st.write(f"**ROUGE-1:** {result['rouge']['rouge1_f']:.4f}")
                st.write(f"**ROUGE-2:** {result['rouge']['rouge2_f']:.4f}")
                st.write(f"**ROUGE-L:** {result['rouge']['rougeL_f']:.4f}")
else:
    st.info("Upload both a PDF and a voice file to proceed.")
