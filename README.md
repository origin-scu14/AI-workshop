# 台北活動推薦 LINE Bot - 台北活動史考特 (Taipei Event Scout）

一個整合台北市開放資料與 AI 的 LINE Bot 系統。  
使用者可透過自然語言查詢台北市最新活動，系統部署於 AWS，透過每日排程同步開放資料並結合 LLM 提供個人化活動推薦。

---

## 功能特色

### 🎯 即時查詢台北市活動
使用者輸入自然語句，例如：「這週末台北有什麼活動？」即可取得近期活動列表。

### 🤖 AI 個人化推薦
採用 LLM（Large Language Model）處理使用者偏好，提供個人化活動推薦，例如展覽、音樂、親子活動等。

### ⭐ 興趣偏好管理
使用者可設定喜好標籤，系統將偏好與查詢紀錄儲存在資料庫，作為排序與推薦依據。

### 🗂 自然語言理解
支援多種查詢方式：  
- 「找信義區明天的活動」  
- 「推薦三個免費的展覽」  
- 「這週末親子可以去哪」  

系統會解析時間、行政區、活動類型等條件。

### 📍 導航功能
活動資訊提供地址與 Google Maps 導航連結。

---

## 技術架構總覽

本系統完全採用 AWS 託管模式，避免混合雲帶來的安全與管理風險，並確保 LINE Webhook 延遲與可用性符合要求。

### 🐍 語言與框架
- 語言：Python  
- Web Framework：FastAPI  
- Lambda Adapter：Mangum（讓 FastAPI 可在 AWS Lambda 上運行）
### ☁ 雲端與運算

- **AWS Lambda**
  - 執行 FastAPI 應用程式（LINE Webhook 後端）
  - 透過 Function URL 或 API Gateway 對外提供 HTTPS 端點

- **Amazon API Gateway（或 Lambda Function URL）**
  - 提供公開 HTTPS Webhook URL，供 LINE Messaging API 呼叫
  - 進行基本路由與安全控管

- **AWS EventBridge / CloudWatch Events**
  - 每日排程觸發「資料同步 Lambda」，將未來 30 天活動從台北開放資料 API 抓取並寫入資料庫

### 🗃 資料庫

- **Amazon DynamoDB**
  - NoSQL Key-Value 資料庫
  - 儲存：
    - 活動資料快取（未來 30 天活動）
    - 使用者基本資料（LINE userId）
    - 使用者偏好標籤
    - 查詢與點擊紀錄
  - 與 AWS 同一雲環境，便於統一帳務與管理

> 說明：資料不再在查詢時即時呼叫政府 API，而是由排程批次同步到 DynamoDB，Bot 僅查詢自己的資料庫，降低延遲。

### 💬 LINE Bot（Messaging API）

- 接收 LINE Webhook 事件
- 使用 reply_token 回覆活動列表與推薦結果
- 若 LLM 推論時間較長，可先回傳「載入中」訊息，再送出最終結果（符合 LINE 需快速 200 OK 的要求）

### 🤖 AI 模組（LLM）

- 使用 LLM API（OpenAI 或 AWS Bedrock ）
- 功能：
  - 意圖辨識（查詢 / 推薦 / 條件變更等）
  - 解析時間、地區、類別等欄位
  - 根據候選活動生成自然語言推薦敘述

> 成本控管：先用 Python + DynamoDB 篩選出符合條件的 5–10 筆活動，再將這少量候選傳給 LLM，而不是把所有活動一次送進模型。

### 🗄 資料來源：台北市開放資料平台

- 台北旅遊網 Open API
  - 活動展覽
  - 活動年曆
- 由「每日同步 Lambda」批次取得未來 30 天活動資料，寫入 DynamoDB 作為查詢來源。

---

## 系統運作流程

### 1. 每日資料同步流程（後台）

1. EventBridge 每日凌晨固定時間觸發「資料同步 Lambda」。
2. 同步 Lambda 呼叫台北開放資料 API，抓取未來 30 天活動列表。
3. 將活動資料寫入 DynamoDB「Events」資料表：
   - eventId
   - title
   - category
   - district
   - startDate / endDate
   - address
   - description
   - imageUrl
4. 完成後紀錄同步時間與筆數（供監控使用）。

> 這個設計避免每位使用者查詢時都直接打政府 API，可降低延遲。

---

### 2. 使用者查詢流程（前台）

1. 使用者在 LINE 輸入訊息，例如：「這週末信義區有什麼親子活動？」  
2. LINE Messaging API 觸發 Webhook，呼叫部署在 API Gateway / Lambda 上的 FastAPI `/callback` 端點。  
3. FastAPI 立即回應 200 OK，並啟動後端處理流程；必要時先回傳「正在為您搜尋活動…」之類的提示訊息。
4. 伺服器呼叫 LLM，將使用者原始文字轉成結構化查詢條件 JSON：
   - dateRange
   - district
   - category
   - count（需要幾筆推薦）
5. 使用查詢條件從 DynamoDB `Events` 表中進行篩選：
   - 日期在目標範圍內
   - 行政區符合（若有）
   - 活動類型符合（若有）
6. 從篩選結果中選出前 N 筆候選（例如 5–10 筆），再送給 LLM 生成自然語言推薦文案與排序。
7. 系統使用 Flex Message 建立活動卡片 Carousel，回傳給使用者。  
8. 同時將查詢條件與使用者對推薦結果的點擊行為寫入 DynamoDB，用於後續偏好分析。  

---

## 專案目錄結構（AWS 版本）

```text
taipei-activity-bot/
│
├── app.py               # FastAPI (LINE Webhook) 主程式 + Mangum handler
├── ai_agent.py          # LLM 意圖解析與推薦文字生成
├── openapi_sync.py      # 每日資料同步 Lambda 腳本
├── dynamodb_service.py  # 與 DynamoDB 互動（活動 & 使用者資料）
├── message_builder.py   # LINE Flex Message 產生器
├── models.py            # 資料模型與查詢條件結構
├── requirements.txt     # 依賴套件
└── template/            # JSON Flex Message 模板（可選）
