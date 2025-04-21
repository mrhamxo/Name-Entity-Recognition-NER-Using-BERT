import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import pandas as pd

# --- Page Settings ---
st.set_page_config(page_title="NER App", layout="wide")
st.title("🧠 Advanced Named Entity Recognition (NER)")
st.markdown("This app uses a fine-tuned BERT model to extract named entities from your input text.")

# --- Sidebar - Label Legend ---
st.sidebar.subheader("🧾 Label Legend")
st.sidebar.markdown("""
- `'1'`: `B-PER` (Beginning of a Person name)
- `'2'`: `I-PER` (Inside of a Person name)
- `'3'`: `B-ORG` (Beginning of an Organization)
- `'4'`: `I-ORG` (Inside of an Organization)
- `'5'`: `B-LOC` (Beginning of a Location)
- `'6'`: `I-LOC` (Inside of a Location)
- `'7'`: `B-MISC` (Beginning of a Miscellaneous entity)
- `'8'`: `I-MISC` (Inside of a Miscellaneous entity)
""")

# --- Load Tokenizer ---
@st.cache_resource
def load_tokenizer(model_path="model/tokenizer"):
    return AutoTokenizer.from_pretrained(model_path)

# --- Load Model ---
@st.cache_resource
def load_model(model_path="model/ner_model"):
    return AutoModelForTokenClassification.from_pretrained(model_path)

# --- Load Pipeline ---
@st.cache_resource
def load_ner_pipeline():
    tokenizer = load_tokenizer()
    model = load_model()
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy=None)

ner = load_ner_pipeline()

# --- Entity ID to Label Map ---
id2label = {
    '1': 'B-PER', '2': 'I-PER',
    '3': 'B-ORG', '4': 'I-ORG',
    '5': 'B-LOC', '6': 'I-LOC',
    '7': 'B-MISC', '8': 'I-MISC'
}

# --- Color map for highlight styling ---
entity_colors = {
    'B-PER': "#DC143C", 'I-PER': "#FFD700",
    'B-ORG': "#228B22", 'I-ORG': "#FF69B4",
    'B-LOC': "#663399", 'I-LOC': "#4169E1",
    'B-MISC': "#FF7F50", 'I-MISC': "#008080"
}

# --- Two Column Layout ---
col1, col2 = st.columns(2)

# --- Input Column ---
with col1:
    st.subheader("✍️ Enter NER Texts:")
    text = st.text_area("", "Bill Gates is the founder of Microsoft.", height=200)

    # --- Predict Button in Column 1 ---
    if st.button("🔍 Run NER"):
        if not text.strip():
            st.warning("Please enter a sentence.")
        else:
            with st.spinner("Analyzing..."):
                results = ner(text)

                # Convert numpy floats to Python float
                for item in results:
                    if isinstance(item["score"], torch.Tensor) or "numpy" in str(type(item["score"])):
                        item["score"] = float(item["score"])

                # Build highlighted annotated text
                highlighted_text = ""
                current_idx = 0
                for r in results:
                    start, end = r["start"], r["end"]
                    token_text = text[start:end]
                    label = r["entity"].replace("LABEL_", "")  # e.g., LABEL_1 => 1
                    label_name = id2label.get(label, "O")
                    color = entity_colors.get(label_name, "#E0E0E0")

                    # Append non-entity text
                    highlighted_text += text[current_idx:start]
                    # Append entity span with color
                    highlighted_text += f'<span style="background-color:{color}; padding:3px 6px; border-radius:4px; margin:2px;">{token_text} <sub><code>{label_name}</code></sub></span>'
                    current_idx = end

                highlighted_text += text[current_idx:]  # Remaining text

            # --- Output Column ---
            with col2:
                st.subheader("✨ Highlighted Output")

                st.markdown("### ")
                st.markdown(highlighted_text, unsafe_allow_html=True)
        
    # --- Summary Table (Outside columns) ---
        st.markdown("### 📊 Entity Summary Table")
        df = pd.DataFrame(results)[["word", "entity", "score", "start", "end"]]
        df["entity"] = df["entity"].apply(lambda e: id2label.get(e.replace("LABEL_", ""), e))
        df["score"] = df["score"].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(df, use_container_width=True)

                

# --- Instructions ---
st.sidebar.subheader("🔍 How to Use:")
st.sidebar.markdown("""
1. **Input Text**: Enter any sentence in the text area to analyze.
2. **Run NER**: Click the "Run NER" button to identify named entities in your text.
3. **Highlighted Output**: Named entities will be displayed with color highlights.
4. **Entity Summary**: A table showing the identified entities, their types, and confidence scores.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 15px;'>
    Developed by <strong>Muhammad Hamza</strong> | Powered by BERT<br><br> 
    <a href='https://github.com/mrhamxo' target='_blank'>
        <img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg' width='30' style='margin-right: 10px;' />
    </a>
    <a href='https://www.linkedin.com/in/muhammad-hamza-khattak/' target='_blank'>
        <img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg' width='30' style='margin-right: 10px;' />
    </a>
    <a href='mailto:mr.hamxa942@gmail.com'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/4/4e/Mail_%28iOS%29.svg' width='30' />
    </a>
</div>
""", unsafe_allow_html=True)