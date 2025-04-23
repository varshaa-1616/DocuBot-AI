import streamlit as st
import pandas as pd
from collections import Counter
from textstat import flesch_kincaid_grade
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
import docx
import PyPDF2
import requests
import google.generativeai as genai
import os
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from dotenv import load_dotenv
load_dotenv()

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# ----------------- INITIAL SESSION STATE -----------------
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""
if "text" not in st.session_state:
    st.session_state.text = None
if "final_summary" not in st.session_state:
    st.session_state.final_summary = None
if "hide_msg" not in st.session_state:
    st.session_state.hide_msg = False
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# ----------------- GEMINI CONFIG -----------------
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-1.5-pro")

# ----------------- LEGAL TERM EXPLAINER -----------------
def explain_legal_term(term):
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{term.lower()}"
        response = requests.get(url)
        data = response.json()
        definition = data[0]['meanings'][0]['definitions'][0]['definition']
        return definition
    except Exception:
        google_search_url = f"https://www.google.com/search?q=legal+term+{term}"
        st.sidebar.markdown(
            f"❌ Couldn't find a definition for **{term}** via the dictionary API.<br>"
            f"<a href='{google_search_url}' target='_blank'>🔎 Search on Google</a>",
            unsafe_allow_html=True
        )
        return None

# ----------------- SIDEBAR CHAT & LEGAL TOOL -----------------
st.sidebar.header("📚 Legal Term Explainer")
query = st.sidebar.text_input("Enter a legal term:")
if query:
    meaning = explain_legal_term(query)
    if meaning:
        st.sidebar.markdown(f"**Definition of '{query}':**")
        st.sidebar.write(meaning)

if st.sidebar.button("💬 Any more queries?"):
    st.session_state.show_chat = not st.session_state.show_chat

# ----------------- MAIN CHATBOT (Regular) -----------------
if st.session_state.show_chat:
    st.sidebar.markdown("### 🤖 Chat Assistant")
    for user, bot in st.session_state.chat_history:
        st.sidebar.markdown(f"**You:** {user}")
        st.sidebar.markdown(f"**Bot:** {bot}")
    st.session_state.chat_input = st.sidebar.text_input("Your message", value=st.session_state.chat_input)
    send_clicked = st.sidebar.button("Send")
    if send_clicked:
        user_input = st.session_state.chat_input.strip()
        if user_input:
            try:
                response = model.generate_content(user_input)
                reply = response.text
            except Exception as e:
                reply = f"⚠️ Error: {str(e)}"
            st.session_state.chat_history.append((user_input, reply))
            st.session_state.chat_input = ""


# ----------------- NLP HELPERS -----------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text

def chunk_text(text, max_words=700):  # Increased chunk size
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# ----------------- CLEANING FUNCTION -----------------
def clean_and_tokenize(text):
    translator = str.maketrans('', '', string.punctuation)
    words = text.lower().translate(translator).split()
    stop_words = set(stopwords.words("english"))
    return [word for word in words if word not in stop_words and word.isalpha()]

def clean_and_tokenize_without_stopwords(text):
    translator = str.maketrans('', '', string.punctuation)
    words = text.lower().translate(translator).split()
    stop_words = set(stopwords.words("english"))
    return [word for word in words if word not in stop_words and word.isalpha()]

# ----------------- MAIN INTERFACE -----------------
st.title("📑 DocuBot AI")
uploaded_file = st.file_uploader("Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])

if uploaded_file:
    if st.session_state.last_uploaded != uploaded_file.name:
        if uploaded_file.name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.name.endswith(".docx"):
            text = extract_text_from_docx(uploaded_file)
        elif uploaded_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()

        # Reset session states
        st.session_state.text = text
        st.session_state.final_summary = None
        st.session_state.hide_msg = False
        st.session_state.last_uploaded = uploaded_file.name
    else:
        text = st.session_state.text

    # Text Stats (keep stopwords in text analysis)
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    paragraph_count = sum(1 for para in text.split('\n') if para.strip() != '')
    character_count = len(text)
    readability_score = flesch_kincaid_grade(text)
    
    # Tokenize and remove stopwords for word cloud and most common words
    cleaned_words = clean_and_tokenize_without_stopwords(text)
    common_words = Counter(cleaned_words).most_common(10)

    if st.button("Summarize Text"):
        try:
            with st.spinner("Summarizing... Please wait."):

                # Increase chunk size to around 700 words per chunk
                chunks = chunk_text(text, max_words=700)
                summaries = []
                total_word_count = 0

                for chunk in chunks:
                    summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
                    total_word_count += len(summary.split())
                    summaries.append(summary)

                final_summary = " ".join(summaries)
                # Ensure summary length is closer to 1300 words
                if total_word_count < 1300:
                    final_summary += " " + summarizer(chunks[-1], max_length=300, min_length=100, do_sample=False)[0]['summary_text']

                st.session_state.final_summary = final_summary
                st.session_state.hide_msg = True
        except Exception as e:
            st.error(f"An error occurred: {e}")

    if st.session_state.final_summary:
        st.subheader("📝 Summary:")
        st.write(st.session_state.final_summary)
        st.download_button("Download Summary", st.session_state.final_summary, file_name="summary.txt")

    st.subheader("📊 Text Analysis")
    st.write(f"Words: {word_count}")
    st.write(f"Sentences: {sentence_count}")
    st.write(f"Paragraphs: {paragraph_count}")
    st.write(f"Characters: {character_count}")
    st.write(f"Readability Score: {readability_score:.2f}")

    st.subheader("🔤 Most Common Words")
    df_common = pd.DataFrame(common_words, columns=["Word", "Count"])
    st.write(df_common)

    st.subheader("☁️ Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(cleaned_words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    stats = {
        "Metric": ["Words", "Sentences", "Paragraphs", "Characters", "Readability Score"],
        "Count": [word_count, sentence_count, paragraph_count, character_count, readability_score]
    }
    df_stats = pd.DataFrame(stats)
    csv = df_stats.to_csv(index=False).encode('utf-8')
    st.download_button("Download Text Stats as CSV", csv, "text_stats.csv", "text/csv")

# ------------------------- 
# 📄 Chatbot (File-Aware)
# -------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Ask based on uploaded file")

if "chat_file_history" not in st.session_state:
    st.session_state.chat_file_history = []

user_file_question = st.sidebar.text_input("Ask about the file...", key="file_chat_input")

if st.sidebar.button("Ask (File Context)"):
    if not st.session_state.text:
        st.sidebar.warning("Please upload a file first!")
    elif user_file_question:
        file_context = st.session_state.text[::]  # Trim if needed
        full_prompt = (
            "Use the following document content to answer the user's question:\n\n"
            f"---\n{file_context}\n---\n\n"
            f"User's question: {user_file_question}"
        )
        try:
            response = model.generate_content(full_prompt)
            st.session_state.chat_file_history.append((user_file_question, response.text))
            st.sidebar.markdown(f"**You:** {user_file_question}")
            st.sidebar.markdown(f"**Bot:** {response.text}")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
