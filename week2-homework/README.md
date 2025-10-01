# 📘 README.md

## 🔧 環境需求
- Python 3.10+  
- OpenAI API 金鑰（可從 [OpenAI](https://platform.openai.com/) 申請）  

---

## 📦 安裝步驟

1. **下載程式碼**  
   把 `app_ui.py` 與 `requirements.txt` 放到一個資料夾中。

2. **建立虛擬環境（建議）**
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
   ```

3. **安裝套件**
   ```bash
   pip install -r requirements.txt
   ```

   如果你沒有 `requirements.txt`，可手動安裝：
   ```bash
   pip install streamlit fastapi uvicorn openai python-dotenv tabulate
   ```

4. **設定 OpenAI API Key**
   在專案目錄建立 `.env` 檔：
   ```
   OPENAI_API_KEY=sk-xxxxxxx
   ```

---

## ▶️ 執行程式

執行：
```bash
streamlit run app_ui.py
```

終端機會顯示網址，例如：
```
Local URL: http://localhost:8501
```

用瀏覽器打開即可。

---

## 💻 使用方法

1. **出題模型**：從下拉選單選擇一個 LLM 來產生題目  
2. **候選模型**：可多選，用來回答題目  
3. **評審模型**：選擇一個 LLM，對所有候選模型的答案進行打分  
4. **題目輸入**：可以手動輸入一個問題（例如「幫 20 人設計 2 天 1 夜台中團建行程」），如果留空，系統會自動由出題模型產生題目  
5. 點擊 **「開始執行」** → 介面會依序顯示：  
   - 📌 題目  
   - 📝 各模型回答（可展開檢視）  
   - 🏆 評審排名表（包含 clarity、argument_strength、overall、rationale）  

---

## 📊 功能總覽
- **自動出題**（旅遊活動領域）
- **多模型回答**（同一題目不同 LLM 給解答）
- **評審打分**（以 clarity + argument_strength 為標準）
- **排名表格**（整體分數排序，並顯示評分理由）
