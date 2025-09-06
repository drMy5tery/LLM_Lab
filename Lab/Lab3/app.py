import streamlit as st
import requests

# ----------------------------------
# Hugging Face API Config
# ----------------------------------
# HF_TOKEN = ""  # Replace this with your Hugging Face API token
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

API_URLS = {
    "T5 (General)": "https://api-inference.huggingface.co/models/t5-base",
    "BioGPT (Healthcare)": "https://api-inference.huggingface.co/models/microsoft/BioGPT-Large",
    "GPT-Neo (Finance-like)": "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-1.3B"
}

# ----------------------------------
# Function to Query API
# ----------------------------------
def query_hf_api(model_url, prompt, is_t5=False):
    inputs = f"question: {prompt}" if is_t5 else prompt
    payload = {"inputs": inputs}

    try:
        response = requests.post(model_url, headers=headers, json=payload, timeout=30)
        output = response.json()

        if isinstance(output, dict) and "error" in output:
            return f"⚠️ Error: {output['error']}"
        elif isinstance(output, list) and "generated_text" in output[0]:
            return output[0]["generated_text"]
        else:
            return f"⚠️ Unexpected API response: {output}"
    except Exception as e:
        return f"❌ Exception: {e}"

# ----------------------------------
# Streamlit App UI
# ----------------------------------
st.set_page_config(page_title="LLM Comparison", layout="centered")
st.title("🤖 Foundation vs Domain-Specific LLMs")
st.markdown("Compare outputs of **T5 (General)**, **BioGPT (Healthcare)**, and **GPT-Neo (Finance-like)** using Hugging Face Inference API.")

# Prompt selection
prompts = {
    "🩺 Healthcare": "What are the symptoms of diabetes?",
    "💰 Finance": "How does inflation affect small investors?",
    "🌿 General": "Explain photosynthesis in simple words."
}

selected_prompt = st.selectbox("📝 Select a prompt:", list(prompts.values()))

if st.button("🚀 Generate Outputs"):
    with st.spinner("Generating output from T5..."):
        t5_output = query_hf_api(API_URLS["T5 (General)"], selected_prompt, is_t5=True)

    with st.spinner("Generating output from BioGPT..."):
        biogpt_output = query_hf_api(API_URLS["BioGPT (Healthcare)"], selected_prompt)

    with st.spinner("Generating output from GPT-Neo..."):
        gptneo_output = query_hf_api(API_URLS["GPT-Neo (Finance-like)"], selected_prompt)

    # -------------------------------
    # Show Model Outputs
    # -------------------------------
    st.subheader("🔍 Results")

    st.markdown("### 🧠 T5 (General)")
    st.success(t5_output)

    st.markdown("### 🏥 BioGPT (Healthcare)")
    st.info(biogpt_output)

    st.markdown("### 💹 GPT-Neo (Finance-like)")
    st.warning(gptneo_output)

    st.markdown("---")
    st.caption("Note: Responses may vary. Powered by Hugging Face Inference API.")
