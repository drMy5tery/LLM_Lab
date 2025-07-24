import streamlit as st
import torch
import re
from transformers import pipeline

# GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def format_story(text):
    """Split into readable sentences."""
    return "\n".join(re.split(r'(?<=[.!?]) +', text.strip()))

@st.cache_resource
def load_models():
    gpt = pipeline("text-generation",
                   model="EleutherAI/gpt-neo-125M",
                   device=0 if DEVICE == "cuda" else -1)

    t5 = pipeline("text2text-generation",
                  model="t5-small",
                  device=-1)  # T5-small on CPU

    return gpt, t5

gpt_pipe, t5_pipe = load_models()

st.set_page_config(page_title="Story Generator (small models)", layout="wide")
st.title("📝 Story Generator with Small Models")
st.markdown("Generates 500+ word stories using **GPT‑Neo 125M (on GPU)** and **T5‑small (on CPU)**")

prompt = st.text_input("Enter your story prompt", "A robot learns to dance under moonlight")
cols = st.columns(4)
temperature = cols[0].slider("Temperature", 0.5, 1.5, 0.9)
top_k = cols[1].slider("Top‑K", 10, 100, 50)
top_p = cols[2].slider("Top‑P", 0.5, 1.0, 0.95)
max_tokens = cols[3].slider("Max Tokens (~500 words = 700)", 300, 900, 700)

if st.button("Generate Stories"):
    with st.spinner("Generating stories..."):
        # GPT‑Neo
        gpt_story = gpt_pipe(prompt,
                             max_new_tokens=max_tokens,
                             temperature=temperature,
                             top_k=top_k,
                             top_p=top_p,
                             do_sample=True,
                             pad_token_id=50256)[0]['generated_text']
        # T5‑small (instruction tuning via prompt prefix)
        t5_prompt = f"write a creative story: {prompt}"
        t5_story = t5_pipe(t5_prompt,
                          max_length=max_tokens,
                          do_sample=True,
                          temperature=temperature,
                          top_k=top_k,
                          top_p=top_p)[0]['generated_text']

    st.subheader("🤖 GPT‑Neo 125M (GPU)")
    st.text_area("Generated Story", value=format_story(gpt_story), height=400)

    st.subheader("🌟 T5‑small (CPU)")
    st.text_area("Generated Story", value=format_story(t5_story), height=400)
