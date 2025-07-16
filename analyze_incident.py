import pandas as pd
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# 设置 Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyChJg8EfcsI2v_GfRxx5s3A9tOWA0F1Quw"

# 1. 读取 CSV
input_file = "incident_raw_11.csv"
df = pd.read_csv(input_file)

# 2. 只保留 Self Caused
self_df = df[df["Causal Type"].str.strip().str.lower() == "self caused"].copy()

# 3. 预处理 Configuration item 得到 app 字段
def extract_app(config_item):
    if pd.isna(config_item):
        return ""
    return str(config_item).split("-")[0].strip()

self_df["app"] = self_df["Configuration item"].apply(extract_app)

# 4. 统计前10 app
app_counts = self_df["app"].value_counts().nlargest(10)
top_apps = set(app_counts.index)
top_df = self_df[self_df["app"].isin(top_apps)].copy()

# 5. LLM 分类
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
prompt = PromptTemplate(
    input_variables=["short_description", "root_cause_description"],
    template=(
        "你是一个IT事件分析专家。请根据以下信息，将事件归类到如下Category之一：FUNCTIONAL WEAKNESS, BUG, Capacity。"
        "并细分Sub category（可选：Data Processing & Validation Issues, System Design & Architecture Limitations, Business Logic Gaps, Capacity & Performance Weaknesses，或你认为合适的其他子类别）。"
        "\n短描述: {short_description}\nRoot cause描述: {root_cause_description}\n"
        "请用如下格式输出：\nCategory: <类别>\nSub category: <子类别>"
    )
)
chain = LLMChain(llm=llm, prompt=prompt)

def classify(row):
    try:
        result = chain.run({
            "short_description": str(row["Short description"]),
            "root_cause_description": str(row["Root cause description"])
        })
        cat_match = re.search(r"Category:\s*(.+)", result)
        subcat_match = re.search(r"Sub category:\s*(.+)", result)
        category = cat_match.group(1).strip() if cat_match else ""
        sub_category = subcat_match.group(1).strip() if subcat_match else ""
        return pd.Series([category, sub_category])
    except Exception as e:
        return pd.Series(["", ""])

# 只处理 top_df
top_df[["Category", "Sub category"]] = top_df.apply(classify, axis=1)

# 6. 写入新 CSV
output_file = "incident_categorized.csv"
top_df.to_csv(output_file, index=False)

print(f"处理完成，结果已保存到 {output_file}") 
