import os
import json
import time
import asyncio
import streamlit as st
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from tabulate import tabulate
from openai import OpenAI

# ─── 讀取環境變數 ───
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

@dataclass
class LLMAnswer:
    provider: str
    model: str
    answer: str

@dataclass
class LLMEval:
    provider: str
    model: str
    clarity: float
    argument_strength: float
    overall: float
    rationale: str

def get_openai() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)

def chat(model: str, system_prompt: str, user_prompt: str,
         temperature: float = 0.3, max_tokens: int = 1200) -> str:
    client = get_openai()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()

# ─── 提示詞 ───
QUESTION_PROMPT = """你是一位嚴謹的出題專家，請提出一個「旅遊活動」的高難度問題，需考驗需求拆解、行程設計與風險備援。只輸出題目本身。"""

ANSWER_PROMPT_TEMPLATE = """題目：
{question}

請提出具體行程方案、風險備援、預算拆解，並以繁體中文回覆。"""

JUDGE_PROMPT_TEMPLATE = """題目：
{question}

候選答案：
{answer}

請依下列指標打分，輸出 JSON：
{{
  "clarity": <0-10>,
  "argument_strength": <0-10>,
  "overall": <0-10>,
  "rationale": "簡短中文說明"
}}"""

def extract_json(text: str) -> dict:
    try:
        s, e = text.find("{"), text.rfind("}")
        return json.loads(text[s:e+1])
    except:
        return {"clarity": 0, "argument_strength": 0, "overall": 0, "rationale": f"解析失敗: {text[:200]}"}

# ─── Streamlit UI ───
st.title("🧪 LLM 出題 / 作答 / 評審 Demo")

# 模型設定
question_setter_model = st.selectbox("出題模型", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
candidate_models = st.multiselect("候選模型", ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"], default=["gpt-4o","gpt-4o-mini"])
judge_model = st.selectbox("評審模型", ["gpt-4o", "gpt-4o-mini"])

# 題目輸入
user_question = st.text_area("題目（留空則自動由出題模型產生）", "")

if st.button("開始執行"):
    t0 = time.time()

    # 1) 出題
    if user_question.strip():
        question = user_question.strip()
    else:
        question = chat(question_setter_model, "你是一位出題專家", QUESTION_PROMPT)

    st.subheader("📌 題目")
    st.write(question)

    # 2) 多模型回答
    answers = []
    for m in candidate_models:
        ans = chat(m, "你是旅遊活動顧問", ANSWER_PROMPT_TEMPLATE.format(question=question), temperature=0.6)
        answers.append(LLMAnswer(provider="openai", model=m, answer=ans))

    st.subheader("📝 各模型回答")
    for a in answers:
        with st.expander(f"{a.model}"):
            st.write(a.answer)

    # 3) 評審
    evals = []
    for a in answers:
        raw = chat(judge_model, "你是嚴謹的評審", JUDGE_PROMPT_TEMPLATE.format(question=question, answer=a.answer))
        data = extract_json(raw)
        evals.append(LLMEval("openai", a.model, float(data["clarity"]), float(data["argument_strength"]), float(data["overall"]), data["rationale"]))

    # 4) 排序結果
    ranking = sorted(evals, key=lambda x: x.overall, reverse=True)
    st.subheader("🏆 排名")
    st.table([[r.model, r.clarity, r.argument_strength, r.overall, r.rationale] for r in ranking])

    st.success(f"完成！耗時 {round(time.time()-t0,2)} 秒")
