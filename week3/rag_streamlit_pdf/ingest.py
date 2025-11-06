
import os
import json
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATA_DIR = Path("./data")
VSTORE_DIR = Path("./vectorstore")
VSTORE_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = VSTORE_DIR / "index.faiss"
DOCS_PATH = VSTORE_DIR / "docs.json"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 中英都好用

def load_pdfs(data_dir: Path) -> List[Dict]:
    docs = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append({
                    "source": pdf_path.name,
                    "page": i + 1,
                    "text": text
                })
    return docs

def chunk_documents(docs: List[Dict], chunk_size=800, chunk_overlap=150) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = []
    for d in docs:
        for chunk in splitter.split_text(d["text"]):
            chunks.append({
                "source": d["source"],
                "page": d["page"],
                "text": chunk.strip()
            })
    return chunks

def build_faiss(chunks: List[Dict]):
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 使用內積，因為已 normalize
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(INDEX_PATH))
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"向量庫完成：{INDEX_PATH} 以及 {DOCS_PATH}")
    print(f"共 {len(chunks)} 個分塊。")

if __name__ == "__main__":
    if not DATA_DIR.exists():
        raise FileNotFoundError("請建立 ./data 資料夾並放入 PDF 檔案")
    raw_docs = load_pdfs(DATA_DIR)
    if not raw_docs:
        raise SystemExit("data 資料夾沒有找到 PDF 檔案。")
    chunks = chunk_documents(raw_docs)
    build_faiss(chunks)
