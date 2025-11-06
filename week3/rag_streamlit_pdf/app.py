
import os
import json
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

# ---------- Config ----------
VSTORE_DIR = Path("./vectorstore")
INDEX_PATH = VSTORE_DIR / "index.faiss"
DOCS_PATH = VSTORE_DIR / "docs.json"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 4

load_dotenv()

@st.cache_resource(show_spinner=False)
def load_embed_and_index():
    if not INDEX_PATH.exists() or not DOCS_PATH.exists():
        st.error("找不到向量庫，請先執行 `python ingest.py` 建置。")
        st.stop()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    index = faiss.read_index(str(INDEX_PATH))
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return model, index, docs

def embed_query(model, q: str) -> np.ndarray:
    vec = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)

def search(index, docs, q_vec: np.ndarray, k: int = TOP_K) -> List[dict]:
    scores, idxs = index.search(q_vec, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        item = docs[int(idx)]
        item = dict(item)  # copy
        item["score"] = float(score)
        results.append(item)
    return results

def build_prompt(question: str, contexts: List[dict]) -> str:
    context_text = "\n\n---\n\n".join([
        f"[來源: {c['source']} p.{c['page']}] \n{c['text']}" for c in contexts
    ])
    prompt = f"""
你是嚴謹的助理。根據下列「提供的資料段落」回答使用者問題。

限制：
- 優先以中文作答。
- 若無法從資料中找到答案，請直接說「我在文件中沒有足夠資訊回答此題」。
- 回答時引用出處，例如 (來源: 檔名 p.頁碼)。

提供的資料段落：
{context_text}

使用者問題：{question}
"""
    return prompt.strip()

def generate_answer(prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return "未設定 OPENAI_API_KEY，無法使用 OpenAI 生成答案。請在環境變數或 .env 設定。"
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"你是專業文件助理。"},
                  {"role":"user","content":prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 PDF RAG Chatbot")
st.caption("把 PDF 轉成向量資料庫（FAISS），透過擷取 + LLM 回答。")

with st.sidebar:
    st.header("步驟")
    st.markdown("1. 把 PDF 放到 `./data`")
    st.markdown("2. 執行 `python ingest.py` 建向量庫")
    st.markdown("3. 這裡輸入問題開始聊天")
    st.divider()
    st.markdown("**設定**")
    TOP_K = st.slider("每次擷取段落數 (Top-K)", 2, 10, 4)

model, index, docs = load_embed_and_index()

question = st.text_input("輸入你的問題（可中英）", placeholder="例如：合約解約流程是什麼？")
go = st.button("送出", type="primary")

if go and question.strip():
    with st.spinner("搜尋向量資料庫中..."):
        q_vec = embed_query(model, question)
        hits = search(index, docs, q_vec, k=TOP_K)

    with st.expander("檢索到的段落 (Top-K)", expanded=False):
        for h in hits:
            st.markdown(f"**{h['source']} p.{h['page']} (score={h['score']:.3f})**")
            st.write(h["text"])

    prompt = build_prompt(question, hits)

    with st.spinner("LLM 生成回答中..."):
        answer = generate_answer(prompt)

    st.subheader("回答")
    st.write(answer)
