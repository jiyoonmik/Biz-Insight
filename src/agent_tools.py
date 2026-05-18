import os
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

load_dotenv()


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def get_pandas_agent(csv_filename, llm=None):
    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_retries=3)
    
    csv_path = os.path.join(DATA_DIR, csv_filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        agent_type="tool-calling",
        allow_dangerous_code=True
    )
    
    return agent

def analyze_company_with_pandas_agent(corp_name: str, agent_type: str) -> str:
    """
    Pandas Dataframe Agent를 이용해 특정 기업의 분석 결과를 반환합니다.
    """
    if agent_type == "financial":
        agent = get_pandas_agent("fs.csv")
        query = f"'{corp_name}' 기업의 재무제표(fs.csv) 데이터를 바탕으로, 수익성과 성장성을 위주로 분석하여 리포트를 작성해줘."
    elif agent_type == "credit":
        agent = get_pandas_agent("credit_data_web.csv")
        query = f"'{corp_name}' 기업의 신용/재무건전성 데이터(credit_data_web.csv)를 바탕으로 신용 위험성과 부채 상환 능력을 분석해줘."
    elif agent_type == "market":
        agent = get_pandas_agent("employee_reviews.csv")
        query = f"'{corp_name}' 기업의 직원 리뷰 데이터(employee_reviews.csv)를 바탕으로 기업 문화와 근무 환경을 분석해줘."
    elif agent_type == "stock":
        agent = get_pandas_agent("investment_data_web.csv")
        query = f"'{corp_name}' 기업의 투자/주식 데이터(investment_data_web.csv)를 바탕으로 현재 가치평가 및 투자 매력도를 분석해줘."
    else:
        return "Unknown agent type"
        
    try:
        response = agent.invoke({"input": query})
        time.sleep(3)  # Rate limit 방지를 위한 의도적 지연
        return response["output"]
    except Exception as e:
        return f"분석 중 오류가 발생했습니다: {str(e)}"
