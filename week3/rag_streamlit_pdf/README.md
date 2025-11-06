
# RAG Chatbot for PDFs (Streamlit + FAISS)

**功能**：
- 將 `./data` 資料夾中的 PDF 讀入、分塊、向量化，並儲存在 `./vectorstore`（FAISS）。
- 以 Streamlit 提供問答介面，從向量資料庫擷取相關段落，並由 LLM（預設 OpenAI）生成答案。

## 快速開始

```bash
# 建議使用虛擬環境
pip install -r requirements.txt

# 匯入 PDF 建置向量庫
python ingest.py  # 會將 ./data/*.pdf 轉成 ./vectorstore/index.faiss 與 docs.json

# 啟動聊天介面
streamlit run app.py
```

## 必要環境變數
- `OPENAI_API_KEY`：若使用 OpenAI 作為生成模型，請設定此變數。

也可在根目錄建立 `.env`：
```
OPENAI_API_KEY=sk-xxxx
```

## 檔案說明
- `ingest.py`：讀取 PDF -> Chunk -> Embedding -> 建 FAISS -> 存檔
- `app.py`：Streamlit 聊天介面（檢索 + 生成）
- `vectorstore/`：向量庫輸出位置
- `data/`：放你的 PDF
