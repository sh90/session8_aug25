# pip install streamlit pandas yfinance
import streamlit as st
from datetime import datetime
from autogen import AssistantAgent, UserProxyAgent, register_function, initiate_chats
import os
import yfinance as yf
from dotenv import load_dotenv
load_dotenv()

print("Imported successfully")
llm_config =  {"model": "gpt-4o-mini","api_key": os.getenv('OPENAI_API_KEY') }

# === Custom Tools ===
def fetch_stock_data(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1mo")
        return {
            "name": info.get("longName", ""),
            "symbol": ticker,
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividends": info.get("dividendRate"),
            "price_to_book": info.get("priceToBook"),
            "debt_to_equity": info.get("debtToEquity"),
            "roe": info.get("returnOnEquity"),
            "prices": hist["Close"].dropna().to_dict(),
        }
    except Exception as e:
        return {"error": str(e)}

# === Streamlit UI ===
st.set_page_config(page_title="Financial Report Generator", layout="wide")
st.title(" ===== Financial Insights Generation with AutoGen ==== ")
ticker = st.text_input("Enter stock ticker Symbol:")
run = st.button("Run Analysis")

# === Agents ===

financial_assistant = AssistantAgent(
    name="FinancialAssistant",
    llm_config=llm_config
)

writer = AssistantAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="""
        You are a professional financial report writer. Based only on the data provided, generate a full markdown-formatted report.
        Include analysis, a data table of key metrics (PE, dividends, ROE, etc.), a summary of the stock prices,
        and suggest future scenarios.

        Return only the markdown content, no explanations or code blocks. 
    """
)

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config=False
)

# === Register tools ===
register_function(
    fetch_stock_data,
    caller=financial_assistant,
    executor=user_proxy,
    name="fetch_stock_data",
    description="Fetch 1-month stock history and key ratios for a given ticker."
)

# === Workflow Execution ===
if run and ticker:
    date_str = datetime.now().strftime("%Y-%m-%d")

    financial_prompt = f"""
        Today is {date_str}. For ticker in {ticker}, call fetch_stock_data(ticker) once and collect the results.
        Return a JSON object summarizing all data. Do not add ```json and ``` in the final output.
    """

    with st.spinner("Fetching financial data..."):
        results = initiate_chats([
            {
                "sender": user_proxy,
                "recipient": financial_assistant,
                "message": financial_prompt,
                "summary_method": "reflection_with_llm",
                "summary_args": {
                    "summary_prompt": "Summarize all financial data and return as a JSON object."
                },
            }
        ])
        print(results)
        data_summary = results[0].summary

        writing_task = f"""
        Use the following financial data to generate the report. 

        {data_summary}

        Generate a markdown financial report including tables, summaries, and future scenarios.
        """

        report_results = initiate_chats([
            {
                "sender": user_proxy,
                "recipient": writer,
                "message": writing_task,
                "summary_method": "last_msg",
                "max_turns": 1
            }
        ])

        final_report = report_results[-1].chat_history[-1]["content"]
        st.markdown("## ===== Final Report ===== ")
        st.markdown(final_report, unsafe_allow_html=True)
