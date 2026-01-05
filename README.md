# LLM Ambedkar 🤖🕊️

**LLM Ambedkar** is a language model interface that simulates intelligent responses and text output in the persona of Dr. B.R. Ambedkar.  
It can be integrated with Unreal Engine or other frontends for interactive AI avatars.

---

## 🚀 Features

- **Text Generation:** Generate context-aware responses based on user prompts.
- **Simple API:** REST-like interface for sending prompts and receiving responses.

## 🛠️ Workflow

1. **Frontend Call:** Unreal Engine or any client sends a text query to the LLM API (`/ask`).  
2. **Text Generation:** LLM processes the query and returns a response.  
> Example:  
> `User → Unreal /ask → LLM → Response`

## 📦 Requirements
All Python dependencies are listed in `requirements.txt`.
Install dependencies using:
```bash
pip install -r requirements.txt
```

## ⚡ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-ambedkar.git
cd llm-ambedkar

# Install dependencies
pip install -r requirements.txt

# Start the API server
python chunks.py
python embed_and_index.py
python search.py
python rag.py
python api.py
uvicorn api:app --reload
```

