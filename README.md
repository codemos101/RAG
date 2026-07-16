# 📄 Simple PDF Q&A System (RAG)

A tiny **RAG** (Retrieval-Augmented Generation) app. Upload a PDF, ask questions,
and get answers grounded in the document — not made up.

This is the Day 7 hands-on project. The code follows the exact steps from the slides:

> **Load → Chunk → Embed → Store** (indexing) **→ Retrieve → Augment → Generate** (querying)

---

## Two versions (pick one)

There are two copies of the app. They are identical except for the **Embed** step:

| File | Embeddings come from | Needs |
|------|---------------------|-------|
| **`app.py`** | Hugging Face `all-MiniLM-L6-v2` (runs locally) | one-time ~80 MB download; no embedding API cost |
| **`app_gemini_embeddings.py`** | Gemini `gemini-embedding-001` (via API) | nothing to download; uses your Gemini key for embeddings too |

- Use **`app.py`** if you want the exact Day-6 model and no API quota worries
  while a whole class indexes PDFs at once. Install: `pip install -r requirements.txt`
- Use **`app_gemini_embeddings.py`** if you want the simplest setup — one API,
  one key, nothing to download. Install: `pip install -r requirements_gemini_embeddings.txt`

Both behave the same once running.

---

## What each tool does

| Tool | Job in the app |
|------|----------------|
| **Streamlit** | The chat web page (the UI) |
| **LangChain** | Glues the pipeline together |
| **PyPDFLoader** | Reads the PDF into text |
| **Sentence Transformers** | Turns text chunks into vectors (embeddings) |
| **FAISS** | Stores the vectors and searches them fast |
| **Google Gemini** | Writes the final answer |

---

## Setup (5 steps)

### 1. Get the files
Put `app.py` and `requirements.txt` in a folder.

### 2. (Recommended) Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install the libraries
```bash
pip install -r requirements.txt
```
> The first run also downloads the small embedding model (~80 MB). This happens
> once, then it's cached.

### 4. Get a free Google Gemini API key
- Go to **https://aistudio.google.com/app/apikey**
- Click **Create API key** and copy it.
- You'll paste it into the app's sidebar (no need to edit any code).

### 5. Run the app
```bash
streamlit run app.py
# ...or the all-Gemini version:
streamlit run app_gemini_embeddings.py
```
Your browser opens automatically. Then:
1. Paste your API key in the sidebar.
2. Upload a PDF.
3. Ask questions!

---

## How to use it in class

1. Upload any PDF (a syllabus, a handbook, a research paper).
2. Ask a question whose answer is **inside** the PDF.
3. Open **"🔍 See the chunks used to answer"** to show students *exactly*
   which pieces of the PDF were retrieved. This makes the invisible
   "retrieval" step visible — the key teaching moment.
4. Try a question whose answer is **NOT** in the PDF. A good RAG system
   says "I don't know" instead of hallucinating.

---

## Tweak these settings (top of `app.py`)

| Setting | Meaning | Try changing to… |
|---------|---------|------------------|
| `CHUNK_SIZE` | characters per chunk | 300 (smaller) or 1000 (bigger) |
| `CHUNK_OVERLAP` | shared characters between chunks | 0 (see context break!) |
| `TOP_K` | how many chunks to retrieve | 1 or 5 |

Changing `CHUNK_OVERLAP` to `0` and asking a question whose answer spans a
chunk boundary is a great way to *show* why overlap matters.

---

## Common problems

- **"No module named ..."** → run `pip install -r requirements.txt` again
  (make sure your virtual environment is active).
- **404 "model not found" OR "no longer available to new users"** → Google
  retires model names over time and sometimes closes older ones to newly-created
  API keys. This app defaults to the stable alias **`gemini-flash-latest`** and
  also shows a **"Gemini model" dropdown in the sidebar** listing the models
  *your key* can actually use. If one name fails, just pick another from the
  dropdown — no code editing needed.
- **API key error** → make sure you copied the whole key and that the
  Gemini free tier is enabled for your Google account.
- **First run is slow** → it's downloading the embedding model once. Later
  runs are fast.
