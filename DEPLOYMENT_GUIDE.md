# Deployment Guide for Streamlit

## Prerequisites

Your app requires:
1. **Data files**: `data/learning_path.db` (SQLite) and `data/chroma/` (vector store)
2. **API keys**: OpenAI (for LLM + embeddings)

---

## Option 1: Streamlit Community Cloud (Recommended - Free)

### Steps:

1. **Push your code to GitHub** (create a new repository)

2. **Add secrets to GitHub repo**:
   - Go to: Settings → Secrets and variables → Actions
   - Add `OPENAI_API_KEY` with your API key

3. **Deploy**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Select your repository
   - Set:
     - Main file path: `app.py`
     - Python version: `3.11`

4. **Important**: For the data files, you'll need to either:
   - **Option A**: Upload them to GitHub (if small enough)
   - **Option B**: Modify the app to fetch data from a cloud storage (S3, etc.)
   - **Option C**: Use Streamlit's secrets management for sensitive data

---

## Option 2: Hugging Face Spaces (Free)

1. Create a Hugging Face account
2. Create a new Space → select "Streamlit" as the SDK
3. Upload your files (or connect to GitHub)
4. Add secrets in Space settings
5. Your app will be available at `https://your-username.hf.space`

---

## Option 3: Render (Free Tier)

1. Push code to GitHub
2. Create a new Web Service on Render
3. Set:
   - Build Command: `pip install -e .`
   - Start Command: `streamlit run app.py --server.port=$PORT`
4. Add environment variables in Render dashboard

---

## Option 4: Railway

1. Connect GitHub to Railway
2. Create a new project → Web Service
3. Set:
   - Build Command: `pip install -e .`
   - Run Command: `streamlit run app.py`
4. Add environment variables

---

## Required Configuration Files

### For any deployment, create `.streamlit/config.toml`:

```toml
[server]
headless = true
port = 8501

[theme]
primaryColor = "#6366F1"
```

### For Streamlit Cloud, create `requirements.txt`:

```
chromadb>=1.4.1
openai>=1.0
pydantic>=2.0
python-dotenv>=1.0
openpyxl>=3.1
pandas>=2.0
langgraph==1.0.8
langchain-openai>=0.2
streamlit>=1.30
langchain-google-genai>=4.2.0
sentence-transformers>=5.2.2
langchain==1.2.10
rank-bm25>=0.2.2
```

---

## Important Notes for Deployment

### 1. Data Files Issue
Your app uses local data files:
- `data/learning_path.db`
- `data/chroma/`

**Solution for cloud deployment:**
- Option A: Add the `data/` folder to GitHub (if files are small)
- Option B: Use cloud storage (AWS S3, Google Cloud Storage)
- Option C: Use a managed database service

### 2. Secrets Management
Add your API key in the deployment platform's secrets/settings, not in code!

### 3. Testing Locally First
Before deploying, test with:
```bash
streamlit run app.py
```

---

## Quick Fix: Include Data in GitHub

If your data files are small enough, add them to GitHub:

```bash
# Edit .gitignore to include data files (remove or comment these lines)
# data/
# !data/.gitkeep
```

Then commit and push the data folder.

---

## Need Help?

For issues with:
- **Data not loading**: Check file paths in deployment
- **API errors**: Verify secrets are set correctly
- **Memory issues**: Some free tiers have memory limits

Let me know which deployment option you'd like to proceed with, and I can provide more specific instructions!
