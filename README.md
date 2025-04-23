

# 📑 DocuBot AI

**DocuBot AI** is an intelligent document assistant that helps users extract key insights, generate summaries, and analyze text data from uploaded `.txt`, `.docx`, or `.pdf` files. It includes features like legal term explanations, chatbot assistance, readability scoring, word cloud generation, and more.

---

## 🚀 Features

- ✅ Upload and analyze `.txt`, `.docx`, or `.pdf` documents  
- 📝 Summarize lengthy documents using HuggingFace transformers  
- 📚 Explain legal terms using dictionary API (with Google fallback)  
- 🤖 Built-in Gemini-powered chatbot for general queries  
- 📊 Text analytics: word, sentence, paragraph, character counts  
- 🔤 Most common words with stopwords removed  
- ☁️ Word Cloud visualization  
- 📈 Flesch-Kincaid Grade readability score  

---

## 🛠️ Tech Stack

- **Frontend & Interface**: [Streamlit](https://streamlit.io)  
- **NLP & Summarization**: `transformers` (DistilBART model)  
- **Readability**: `textstat`  
- **Legal Definitions**: [Free Dictionary API](https://dictionaryapi.dev/)  
- **Word Cloud**: `wordcloud`, `matplotlib`  
- **PDF/DOCX Parsing**: `PyPDF2`, `python-docx`  
- **Chatbot**: Google Gemini 1.5 Pro (via Generative AI API)  
- **Environment Management**: `python-dotenv`

---

## 📂 Project Structure

```plaintext
├── app.py               # Main Streamlit app
├── requirements.txt     # Dependencies
├── .env                 # API keys
└── README.md            # This file
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/varshaa-1616/docubot-ai.git
   cd docubot-ai
   ```

2. **Create and activate a virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables**  
   Create a `.env` file:
   ```env
   API_KEY=your_google_generative_ai_key
   ```

5. **Run the application**  
   ```bash
   streamlit run app.py
   ```

---

## 🧪 Example Use Cases

- Summarizing legal or academic PDFs  
- Understanding dense legal jargon with dictionary assistance  
- Analyzing document complexity and readability  
- Exploring common terms via frequency tables and word clouds  
- Quick chatbot queries within the sidebar

