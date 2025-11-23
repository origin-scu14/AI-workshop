# 台北活動推薦 LINE Bot（活動史考特 Taipei Event Scout）

一個整合台北市開放資料與 AI 的 LINE Bot 系統。使用者可透過自然語言查詢台北市最新活動，系統使用 Hugging Face Spaces 作為後端，整合台北開放資料平台並提供個人化活動推薦。

---

## 功能特色

### 🎯 即時查詢台北市活動  
使用者輸入自然語句，例如：「這週末台北有什麼活動？」即可取得近期活動列表。

### 🤖 AI 個人化推薦  
採用 LLM（Large Language Model）處理使用者偏好，提供個人化活動推薦，例如展覽、音樂、親子活動等。

### ⭐ 興趣偏好管理  
使用者可設定喜好標籤，並由 Firestore 儲存。系統會依偏好調整排序結果。

### 🗂 自然語言理解  
支援多種查詢方式：  
- 「找信義區明天的活動」  
- 「推薦三個免費的展覽」  
- 「這週末親子可以去哪」  

系統會自動解析時間、行政區、活動類型等資訊。

### 📍 導航功能  
活動資訊提供地址與 Google Maps 導航連結。

---

## 技術架構

### 🐍 語言  
- Python（後端主要執行環境）

### ☁ 部署平台：Hugging Face Spaces  
後端採用 Hugging Face Spaces 執行 FastAPI：  
- 提供 LINE Webhook 所需的 HTTPS 公開端點  
- 執行 LLM 解析  
- 呼叫台北開放資料 API  
- 連接 Firebase Firestore  

### 💬 LINE Bot（Messaging API）  
負責：  
- 處理使用者訊息  
- 回傳活動查詢結果  
- 回覆 Flex Message 卡片  

### 🤖 AI 模組（LLM）  
使用 LLM API（OpenAI、Hugging Face Inference API 任一）負責：  
- 意圖辨識  
- 時間、地區、類型解析  
- 活動摘要生成  
- 推薦排序的文字生成  

### 🗄 資料來源：台北市開放資料平台  
統一透過台北旅遊網 Open API 取得活動資訊，包含：  
- 活動展覽  
- 活動年曆  
- 類別、地點、時間、介紹等欄位

### 📂 Firebase Firestore  
Firestore 用途：  
- 儲存使用者 ID  
- 活動偏好標籤  
- 查詢紀錄  
- 推薦排序所需的行為資料

---

## 系統運作流程

1. 使用者在 LINE 中輸入查詢訊息  
2. LINE 平台將 Webhook 請求轉送至 Hugging Face Space  
3. FastAPI 分析訊息內容  
4. AI 模組（LLM）解析意圖、時間、地點、分類等條件  
5. 後端透過台北開放資料 API 取得各項活動資訊  
6. 若使用者有偏好設定，從 Firestore 取回偏好資料  
7. 統合排序後，產生 Flex Message 活動卡片  
8. LINE Bot 回傳活動推薦  
9. 系統將查詢紀錄與互動資料回存 Firestore

## 技術棧

- Python  
- FastAPI  
- LINE Messaging API  
- Hugging Face Spaces  
- Firebase Firestore  
- 台北市開放資料 API  
- LLM API（OpenAI / Hugging Face）

