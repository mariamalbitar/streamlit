import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
from llama_index.llms.openrouter import OpenRouter
from llama_index.core import Settings, Document, StorageContext, load_index_from_storage
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.indices.struct_store import JSONQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModel
import logging
import time
from typing import Dict, Optional
import torch

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Disable proxy settings
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# Define stock schema
stock_schema = {
    "type": "object",
    "properties": {
        "Apple": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Date": {"type": "string"},
                    "Close": {"type": "number"},
                    "Open": {"type": "number"},
                    "High": {"type": "number"},
                    "Low": {"type": "number"},
                    "Volume": {"type": "integer"}
                },
                "required": ["Date", "Close"]
            }
        },
        "Meta": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Date": {"type": "string"},
                    "Close": {"type": "number"},
                    "Open": {"type": "number"},
                    "High": {"type": "number"},
                    "Low": {"type": "number"},
                    "Volume": {"type": "integer"}
                },
                "required": ["Date", "Close"]
            }
        },
        "Microsoft": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Date": {"type": "string"},
                    "Close": {"type": "number"},
                    "Open": {"type": "number"},
                    "High": {"type": "number"},
                    "Low": {"type": "number"},
                    "Volume": {"type": "integer"}
                },
                "required": ["Date", "Close"]
            }
        }
    }
}

# Cache language model
@st.cache_resource
def load_language_model():
    start_time = time.time()
    try:
        os.environ["OPENROUTER_API_KEY"] = "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"
        llm = OpenRouter(
            api_key=os.environ["OPENROUTER_API_KEY"],
            model="mistralai/mixtral-8x7b-instruct",
            max_tokens=512,
            context_window=4096,
        )
        Settings.llm = llm
        Settings.chunk_size = 1024
        logging.info(f"Language model loaded in {time.time() - start_time} seconds")
        return llm
    except Exception as e:
        logging.error(f"Error initializing language model: {e}")
        st.error(f"Error initializing language model: {e}")
        return None

# Cache embedding model
@st.cache_resource
def load_embedding_model():
    start_time = time.time()
    try:
        model_name = "BAAI/bge-small-en-v1.5"
        cache_dir = r"C:\Users\Fa\Desktop\Streamlit-Authentication-main\model_cache"
        os.makedirs(cache_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        AutoModel.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        embed_model = HuggingFaceEmbedding(model_name=model_name, cache_folder=cache_dir, device=device)
        Settings.embed_model = embed_model
        logging.info(f"Embedding model loaded in {time.time() - start_time} seconds")
        return embed_model
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}")
        st.warning(f"Failed to load embedding model: {e}. Visualizations will be based on available data.")
        return None

# Cache financial data loading
@st.cache_data
def load_financial_data_cached(file_path: str, company_name: str) -> pd.DataFrame:
    start_time = time.time()
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            logging.error(f"File not found: {file_path}")
            return pd.DataFrame()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if isinstance(raw_data, dict):
            if company_name in raw_data:
                df = pd.DataFrame(raw_data[company_name])
            else:
                for key in raw_data.keys():
                    if key.lower() == company_name.lower():
                        df = pd.DataFrame(raw_data[key])
                        break
                else:
                    first_key = next(iter(raw_data), None)
                    if first_key:
                        df = pd.DataFrame(raw_data[first_key])
                    else:
                        raise ValueError("No valid key found in JSON file.")
        else:
            df = pd.DataFrame(raw_data)
        
        if len(df.columns) == 1 and df.columns[0] == 0:
            df = df.rename(columns={0: 'RawText'})
        logging.info(f"Financial data for {company_name} loaded in {time.time() - start_time} seconds")
        return df
    except json.JSONDecodeError as e:
        st.warning(f"Failed to load file as JSON: {e}. Attempting to read as text...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            data = []
            for line in lines:
                line = line.strip()
                if line:
                    sentiment = line.split('@')[-1] if '@' in line else None
                    text = line.split('@')[0].strip() if '@' in line else line
                    data.append({"Sentence": text, "Sentiment": sentiment})
            df = pd.DataFrame(data)
            logging.info(f"Financial data for {company_name} loaded as text in {time.time() - start_time} seconds")
            return df
        except Exception as e:
            st.error(f"Failed to read file as text: {e}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load stock data
def load_stock_data():
    companies = {
        "Apple": r"C:\Users\Fa\Desktop\Streamlit-Authentication-main\stock_AAPL-1.json",
        "Meta": r"C:\Users\Fa\Desktop\Streamlit-Authentication-main\stock_META-1.json",
        "Microsoft": r"C:\Users\Fa\Desktop\Streamlit-Authentication-main\stock_MSFT-1.json",
    }
    apple_data = load_financial_data_cached(companies["Apple"], "Apple")
    meta_data = load_financial_data_cached(companies["Meta"], "Meta")
    msft_data = load_financial_data_cached(companies["Microsoft"], "Microsoft")
    return apple_data, meta_data, msft_data

# Cache phrasebank data
@st.cache_data
def load_phrase_data():
    start_time = time.time()
    file_path = r"C:\Users\Fa\Desktop\Streamlit-Authentication-main\financial_phrasebank (2).json"
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            logging.error(f"File not found: {file_path}")
            return None, pd.DataFrame()
        with open(file_path, "r", encoding="utf-8") as f:
            phrase_data = json.load(f)
        phrase_df = pd.DataFrame(phrase_data)
        if len(phrase_df.columns) == 1:
            sentences = phrase_df.iloc[:, 0]
            split_sentences = sentences.str.rsplit('@', n=1, expand=True)
            split_sentences.columns = ['Sentence', 'Sentiment']
            phrase_df = split_sentences
        logging.info(f"Phrasebank data loaded in {time.time() - start_time} seconds")
        return phrase_data, phrase_df
    except Exception as e:
        st.warning(f"Failed to load financial_phrasebank: {e}")
        logging.error(f"Failed to load {file_path}: {e}")
        return None, pd.DataFrame()

# Cache stage data
@st.cache_data
def load_stage_data():
    start_time = time.time()
    file_path = r"C:\Users\Fa\Desktop\Streamlit-Authentication-main\cleaned.json"
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            logging.error(f"File not found: {file_path}")
            return None, pd.DataFrame()
        with open(file_path, "r", encoding="utf-8") as f:
            stage_data = json.load(f)
        stage_df = pd.DataFrame(stage_data)
        logging.info(f"Stage data loaded from {file_path} in {time.time() - start_time} seconds")
        logging.debug(f"Stage data sample: {stage_data[:2] if stage_data else 'Empty'}")
        return stage_data, stage_df
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON in {file_path}: {e}")
        logging.error(f"JSON decode error in {file_path}: {e}")
        return None, pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load {file_path}: {e}")
        logging.error(f"Error loading {file_path}: {e}")
        return None, pd.DataFrame()

# Cache query engines
@st.cache_resource
def setup_query_engines(apple_data, meta_data, msft_data, phrase_data, stage_data):
    start_time = time.time()
    tools = []
    # Apple data
    if not apple_data.empty:
        apple_engine = JSONQueryEngine(json_value={"Apple": apple_data.to_dict(orient="records")}, json_schema=stock_schema)
        tools.append(QueryEngineTool.from_defaults(
            query_engine=apple_engine,
            name="Apple_Financials",
            description="Use this tool to query Apple's financial data, such as stock prices."
        ))
    # Meta data
    if not meta_data.empty:
        meta_engine = JSONQueryEngine(json_value={"Meta": meta_data.to_dict(orient="records")}, json_schema=stock_schema)
        tools.append(QueryEngineTool.from_defaults(
            query_engine=meta_engine,
            name="Meta_Financials",
            description="Use this tool to query Meta's financial data, such as stock prices."
        ))
    # Microsoft data
    if not msft_data.empty:
        msft_engine = JSONQueryEngine(json_value={"Microsoft": msft_data.to_dict(orient="records")}, json_schema=stock_schema)
        tools.append(QueryEngineTool.from_defaults(
            query_engine=msft_engine,
            name="Microsoft_Financials",
            description="Use this tool to query Microsoft's financial data, such as stock prices."
        ))
    # Phrasebank data
    if Settings.embed_model is not None and phrase_data:
        try:
            persist_dir = r"C:\Users\Fa\Desktop\Streamlit-Authentication-main\index_storage"
            if os.path.exists(persist_dir):
                storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                phrase_index = load_index_from_storage(storage_context)
            else:
                phrase_docs = [Document(text=str(item)) for item in phrase_data]
                phrase_index = VectorStoreIndex.from_documents(phrase_docs)
                phrase_index.storage_context.persist(persist_dir=persist_dir)
            phrase_engine = phrase_index.as_query_engine(similarity_top_k=3)
            tools.append(QueryEngineTool.from_defaults(
                query_engine=phrase_engine,
                name="Phrasebank_Tool",
                description="Use this tool to search financial phrases and analyze sentiment."
            ))
        except Exception as e:
            st.warning(f"Error setting up financial phrase engine: {e}")
            logging.error(f"Error setting up phrase engine: {e}")
    # Stage data
    if stage_data is not None:
        try:
            stage_schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "stage": {"type": "string"},
                        "updated_at": {"type": "string"}
                    },
                    "required": ["company", "stage", "updated_at"]
                }
            }
            stage_engine = JSONQueryEngine(json_value=stage_data, json_schema=stage_schema)
            tools.append(QueryEngineTool.from_defaults(
                query_engine=stage_engine,
                name="Stage_Tool",
                description="Use this tool to query company maturity stages."
            ))
        except Exception as e:
            st.warning(f"Error setting up maturity stage engine: {e}")
            logging.error(f"Error setting up stage engine: {e}")
    logging.info(f"Query engines setup in {time.time() - start_time} seconds")
    return tools

# Cache router engine
@st.cache_resource
def setup_router_engine(_tools, _llm):
    start_time = time.time()
    try:
        if _llm and _tools:
            selector = LLMSingleSelector.from_defaults(llm=_llm)
            router_engine = RouterQueryEngine.from_defaults(selector=selector, query_engine_tools=_tools, llm=_llm)
            logging.info(f"Router engine setup in {time.time() - start_time} seconds")
            return router_engine
        else:
            st.warning("Router engine disabled due to failure in initializing language model or tools.")
            return None
    except Exception as e:
        st.error(f"Error initializing router engine: {e}")
        logging.error(f"Error initializing router engine: {e}")
        return None

# Function for natural language queries
def ask_router(query: str, router_engine) -> str:
    if router_engine is None:
        return "Router engine not initialized due to an error."
    try:
        response = router_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error processing query: {e}"

# Visualization functions
def plot_stock_data(df: pd.DataFrame, company_name: str):
    if 'Date' in df.columns and 'Close' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date')
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Close'], color='#26c6da', label='Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{company_name} Closing Prices')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.warning(f"No 'Date' or 'Close' column found in {company_name} data.")

def plot_stock_comparison(stock_dfs: Dict[str, pd.DataFrame]):
    plt.figure(figsize=(14, 7))
    has_data = False
    for company, df in stock_dfs.items():
        if 'Date' in df.columns and 'Close' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).sort_values('Date')
            plt.plot(df['Date'], df['Close'], marker='o', label=company)
            has_data = True
    if has_data:
        plt.title('Stock Closing Price Comparison')
        plt.xlabel('Date')
        plt.ylabel('Closing Price (USD)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.warning("No valid stock data available for comparison.")

def plot_sentiment_data(df: pd.DataFrame, title: str = "Sentiment Distribution"):
    if 'Sentiment' in df.columns:
        sentiment_counts = df['Sentiment'].value_counts()
        plt.figure(figsize=(7, 7))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                startangle=140, colors=['#26c6da', '#66bb6a', '#ef5350'])
        plt.title(title)
        st.pyplot(plt)
    else:
        st.warning("No 'Sentiment' column found in the data.")

def plot_stage_data(df: pd.DataFrame, title: str = "Maturity Stage Distribution"):
    if 'stage' in df.columns:
        stage_counts = df['stage'].value_counts().sort_index()
        plt.figure(figsize=(8, 5))
        plt.bar(stage_counts.index.astype(str), stage_counts.values, color='#ab47bc')
        plt.title(title)
        plt.xlabel('Stage')
        plt.ylabel('Number of Companies')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.warning("No 'stage' column found in the data.")

def plot_dpd_data(df: pd.DataFrame, title: str = "Days Past Due (DPD) Distribution"):
    if 'DPD' in df.columns:
        plt.figure(figsize=(12, 5))
        plt.hist(df['DPD'].dropna(), bins=20, color='#ef5350', edgecolor='black')
        plt.title(title)
        plt.xlabel('Days Past Due (DPD)')
        plt.ylabel('Number of Companies')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.warning("No 'DPD' column found in the data.")

def plot_credit_expiration(df: pd.DataFrame, title: str = "Credit Expiration Distribution"):
    if 'Credit Expiration' in df.columns:
        plt.figure(figsize=(12, 5))
        plt.hist(df['Credit Expiration'].dropna(), bins=20, color='#26c6da', edgecolor='black')
        plt.title(title)
        plt.xlabel('Credit Expiration (Days)')
        plt.ylabel('Number of Companies')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.warning("No 'Credit Expiration' column found in the data.")