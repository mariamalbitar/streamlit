import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from llama_index.llms.openrouter import OpenRouter
from llama_index.core import Settings, Document
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.indices.struct_store import JSONQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


st.set_page_config(page_title="Financial Insights Dashboard", layout="wide")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
.stApp {
    background-color: #F3F4F6;
    font-family: 'Roboto', sans-serif;
}
.auth-container {
    background: #FFFFFF;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    max-width: 400px;
    margin: 3rem auto;
    text-align: center;
}
.main-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    color: #1E3A8A;
    margin-bottom: 1rem;
}
.logo {
    display: block;
    margin: 0 auto 1rem;
    max-width: 100px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    justify-content: center;
}
.stTabs [data-baseweb="tab"] {
    background-color: #FFFFFF;
    color: #1E3A8A;
    font-weight: 400;
    font-size: 1rem;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    transition: all 0.3s ease;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #1E3A8A;
    color: #FFFFFF;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #E5E7EB;
}
.stTextInput > div > div > input {
    background-color: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 8px;
    padding: 0.7rem;
    color: #1E3A8A;
    font-size: 0.95rem;
}
.stTextInput > div > div > input:focus {
    border-color: #1E3A8A;
    box-shadow: 0 0 4px rgba(30, 58, 138, 0.3);
}
.stButton > button {
    background-color: #1E3A8A;
    color: #FFFFFF;
    border-radius: 8px;
    padding: 0.7rem 1.5rem;
    font-weight: 400;
    font-size: 0.95rem;
    transition: all 0.3s ease;
    width: 100%;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
.stButton > button:hover {
    background-color: #3B82F6;
    transform: scale(1.03);
}
.stAlert {
    border-radius: 8px;
    padding: 0.8rem;
    font-size: 0.9rem;
}
.company-data {
    background: #FFFFFF;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    margin: 2rem 0;
}
.kpi-card {
    background: #FFFFFF;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    padding: 1rem;
    margin: 0.5rem;
    text-align: center;
}
.kpi-card h3 {
    margin: 0;
    font-size: 0.9rem;
    color: #1E3A8A;
}
.kpi-card p {
    margin: 0.5rem 0 0;
    font-size: 1.2rem;
    font-weight: 700;
}
.positive {
    color: #10B981;
}
.negative {
    color: #EF4444;
}
.sidebar .sidebar-content {
    background-color: #1E3A8A;
    color: #FFFFFF;
}
.sidebar h2, .sidebar p {
    color: #FFFFFF;
}
.logout-button {
    background-color: #EF4444;
    color: #FFFFFF;
    border-radius: 8px;
    padding: 0.7rem;
    font-weight: 400;
    font-size: 0.95rem;
    transition: all 0.3s ease;
    width: 100%;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
.logout-button:hover {
    background-color: #DC2626;
    transform: scale(1.03);
}
.footer {
    text-align: center;
    color: #6B7280;
    font-size: 0.85rem;
    margin-top: 2rem;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">', unsafe_allow_html=True)


API_URL = "http://127.0.0.1:8002"


if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None


companies_paths = {
    "Apple": r"C:\Users\Fa\Desktop\Streamlit-Authentication-main\stock_AAPL-1.json",
    "Meta": r"C:\Users\Fa\Desktop\Streamlit-Authentication-main\stock_META-1.json",
    "Microsoft": r"C:\Users\Fa\Desktop\Streamlit-Authentication-main\stock_MSFT-1.json",
    "cleaned": r"C:\Users\Fa\Desktop\Streamlit-Authentication-main\cleaned.json",
    "financial_phrasebank": r"C:\Users\Fa\Desktop\Streamlit-Authentication-main\financial_phrasebank (2).json"
}
companies = ['Apple', 'Meta', 'Microsoft']


REQUIRED_STOCK_COLUMNS = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']


os.environ["OPENROUTER_API_KEY"] = "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"
llm = OpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="mistralai/mixtral-8x7b-instruct",
    max_tokens=512,
    context_window=4096,
)
Settings.llm = llm
Settings.chunk_size = 1024
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model


try:
    with open(companies_paths["financial_phrasebank"], "r", encoding="utf-8") as f:
        phrase_data = json.load(f)
    phrase_docs = [Document(text=item) for item in phrase_data]
    phrase_index = VectorStoreIndex.from_documents(phrase_docs)
    phrase_engine = phrase_index.as_query_engine(similarity_top_k=3)
except Exception as e:
    st.error(f"Error loading financial_phrasebank data: {e}")
    phrase_engine = None


try:
    with open(companies_paths["cleaned"], "r", encoding="utf-8") as f:
        stage_data = json.load(f)
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
except Exception as e:
    st.error(f"Error loading cleaned data: {e}")
    stage_engine = None


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

try:
    apple_data = json.load(open(companies_paths["Apple"], "r", encoding="utf-8"))
    meta_data = json.load(open(companies_paths["Meta"], "r", encoding="utf-8"))
    msft_data = json.load(open(companies_paths["Microsoft"], "r", encoding="utf-8"))
    apple_engine = JSONQueryEngine(json_value=apple_data, json_schema=stock_schema)
    meta_engine = JSONQueryEngine(json_value=meta_data, json_schema=stock_schema)
    msft_engine = JSONQueryEngine(json_value=msft_data, json_schema=stock_schema)
except Exception as e:
    st.error(f"Error loading stock data: {e}")
    apple_engine, meta_engine, msft_engine = None, None, None


tools = [
    QueryEngineTool.from_defaults(
        query_engine=apple_engine,
        name="Apple_Financials",
        description="Use this for questions about Apple's financial data."
    ) if apple_engine else None,
    QueryEngineTool.from_defaults(
        query_engine=meta_engine,
        name="Meta_Financials",
        description="Use this for questions about Meta's financial data."
    ) if meta_engine else None,
    QueryEngineTool.from_defaults(
        query_engine=msft_engine,
        name="Microsoft_Financials",
        description="Use this for questions about Microsoft's financial data."
    ) if msft_engine else None,
    QueryEngineTool.from_defaults(
        query_engine=phrase_engine,
        name="Phrasebank_Tool",
        description="Use this to search phrases in the financial phrasebank."
    ) if phrase_engine else None,
    QueryEngineTool.from_defaults(
        query_engine=stage_engine,
        name="Stage_Tool",
        description="Use this for questions about company maturity stages."
    ) if stage_engine else None,
]
tools = [tool for tool in tools if tool is not None]  # Remove None tools
if tools:
    selector = LLMSingleSelector.from_defaults(llm=llm)
    router_engine = RouterQueryEngine.from_defaults(
        selector=selector,
        query_engine_tools=tools,
        llm=llm
    )
else:
    router_engine = None
    st.error("No valid query engines available.")


def load_financial_data(file_path, company_name):
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}. Using sample data.")
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        if isinstance(raw_data, dict) and company_name in raw_data:
            df = pd.DataFrame(raw_data[company_name])
        else:
            df = pd.DataFrame(raw_data)
        
        missing_columns = [col for col in REQUIRED_STOCK_COLUMNS if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns in {file_path}: {', '.join(missing_columns)}. Using sample data.")
            return None
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            if df['Date'].isna().all():
                st.error(f"Invalid date format in {file_path}. Using sample data.")
                return None
            df = df.dropna(subset=['Date']).sort_values('Date')
        return df
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}. Using sample data.")
        return None


def load_cleaned_data():
    try:
        response = requests.get(f"{API_URL}/data/cleaned", proxies={"http": None, "https": None})
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        required_columns = ['Credit Expiration', 'Current Stage']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns in cleaned data: {', '.join(missing_columns)}.")
            return None
        return df
    except Exception as e:
        st.warning(f"Error fetching cleaned data: {e}. Using local data.")
        return load_financial_data(companies_paths['cleaned'], 'cleaned')


def load_phrasebank_data():
    try:
        response = requests.get(f"{API_URL}/data/financial_phrasebank", proxies={"http": None, "https": None})
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=['Text'])
        df['Sentiment'] = df['Text'].str.extract(r'@(\w+)$')
        df['Text'] = df['Text'].str.replace(r'@\w+$', '', regex=True)
        if df['Sentiment'].isna().any():
            st.error("Some entries in financial_phrasebank data are missing sentiment labels.")
            return None
        return df
    except Exception as e:
        st.warning(f"Error fetching financial_phrasebank data: {e}. Using local data.")
        return load_financial_data(companies_paths['financial_phrasebank'], 'financial_phrasebank')

sample_data = {
    'Apple': pd.DataFrame({
        'Date': pd.date_range(start='2025-01-01', end='2025-07-12', freq='D'),
        'Close': [150 + i * 0.3 + (i % 8) for i in range(193)],
        'Open': [148 + i * 0.3 + (i % 8) for i in range(193)],
        'High': [152 + i * 0.3 + (i % 8) for i in range(193)],
        'Low': [146 + i * 0.3 + (i % 8) for i in range(193)],
        'Volume': [2000000 + i * 2000 for i in range(193)]
    }),
    'Meta': pd.DataFrame({
        'Date': pd.date_range(start='2025-01-01', end='2025-07-12', freq='D'),
        'Close': [300 + i * 0.5 + (i % 10) for i in range(193)],
        'Open': [298 + i * 0.5 + (i % 10) for i in range(193)],
        'High': [302 + i * 0.5 + (i % 10) for i in range(193)],
        'Low': [296 + i * 0.5 + (i % 10) for i in range(193)],
        'Volume': [1000000 + i * 1000 for i in range(193)]
    }),
    'Microsoft': pd.DataFrame({
        'Date': pd.date_range(start='2025-01-01', end='2025-07-12', freq='D'),
        'Close': [250 + i * 0.4 + (i % 7) for i in range(193)],
        'Open': [248 + i * 0.4 + (i % 7) for i in range(193)],
        'High': [252 + i * 0.4 + (i % 7) for i in range(193)],
        'Low': [246 + i * 0.4 + (i % 7) for i in range(193)],
        'Volume': [1500000 + i * 1500 for i in range(193)]
    })
}


data = {}
for company, path in companies_paths.items():
    if company in companies:
        df = load_financial_data(path, company)
        data[company] = df if df is not None else sample_data[company]
    elif company == 'cleaned':
        data[company] = load_cleaned_data()
    elif company == 'financial_phrasebank':
        data[company] = load_phrasebank_data()


def login_page():
    st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
    st.markdown("<h2><i class='fas fa-sign-in-alt'></i> Log In</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submit = st.form_submit_button("Log In")
            if submit:
                if username.strip() and password.strip():
                    with st.spinner("Logging in..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/login",
                                data={"username": username.strip(), "password": password.strip()},
                                proxies={"http": None, "https": None}
                            )
                            if response.status_code == 200:
                                st.session_state.logged_in = True
                                st.session_state.username = username.strip()
                                st.success(response.json().get("msg", "Login successful!"))
                                st.rerun()
                            else:
                                st.error(response.json().get("detail", "Invalid credentials"))
                        except Exception as e:
                            st.error(f"Failed to connect to server: {e}")
                else:
                    st.warning("Please fill in all fields.")
    st.markdown("<p>Don't have an account? <a href='#' onclick='st.session_state.page=\"signup\";st.rerun()'>Sign Up</a></p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def signup_page():
    st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
    st.markdown("<h2><i class='fas fa-user-plus'></i> Sign Up</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("signup_form"):
            username = st.text_input("Username", placeholder="Choose a username")
            email = st.text_input("Email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Choose a password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password")
            submit = st.form_submit_button("Sign Up")
            if submit:
                if username.strip() and email.strip() and password.strip() and confirm_password.strip():
                    if password == confirm_password:
                        with st.spinner("Registering..."):
                            try:
                                response = requests.post(
                                    f"{API_URL}/register",
                                    data={"username": username.strip(), "password": password.strip()},
                                    proxies={"http": None, "https": None}
                                )
                                if response.status_code == 200:
                                    st.session_state.logged_in = True
                                    st.session_state.username = username.strip()
                                    st.success(response.json().get("msg", "Registration successful!"))
                                    st.rerun()
                                else:
                                    st.error(response.json().get("detail", "Registration failed"))
                            except Exception as e:
                                st.error(f"Failed to connect to server: {e}")
                    else:
                        st.error("Passwords do not match.")
                else:
                    st.warning("Please fill in all fields.")
    st.markdown("<p>Already have an account? <a href='#' onclick='st.session_state.page=\"login\";st.rerun()'>Log In</a></p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def visualize_cleaned_data(df):
    if df is None or df.empty:
        st.error("No data available for Cleaned Data.")
        return
    st.markdown("<h2>Cleaned Data Analysis</h2>", unsafe_allow_html=True)
    
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Average Credit Expiration</h3>", unsafe_allow_html=True)
        avg_credit = df['Credit Expiration'].mean()
        st.markdown(f"<p>{avg_credit:.2f} days</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Stage 1 Count</h3>", unsafe_allow_html=True)
        stage1_count = len(df[df['Current Stage'] == 1])
        st.markdown(f"<p>{stage1_count}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Stage 2 Count</h3>", unsafe_allow_html=True)
        stage2_count = len(df[df['Current Stage'] == 2])
        st.markdown(f"<p>{stage2_count}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    
    st.markdown("<h3>Credit Expiration Distribution</h3>", unsafe_allow_html=True)
    fig_credit = px.histogram(df, x='Credit Expiration', nbins=20, title="Credit Expiration Days")
    fig_credit.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Roboto", size=12, color="#1E3A8A"),
        xaxis_title="Credit Expiration (Days)",
        yaxis_title="Count"
    )
    st.plotly_chart(fig_credit, use_container_width=True)

    
    st.markdown("<h3>Days Past Due (DPD) Distribution</h3>", unsafe_allow_html=True)
    fig_dpd = px.histogram(df, x='DPD', nbins=20, title="Days Past Due (DPD)", color_discrete_sequence=['salmon'])
    fig_dpd.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Roboto", size=12, color="#1E3A8A"),
        xaxis_title="DPD",
        yaxis_title="Number of Customers"
    )
    st.plotly_chart(fig_dpd, use_container_width=True)

    
    st.markdown("<h3>Stage Distribution</h3>", unsafe_allow_html=True)
    stage_counts = df['Current Stage'].value_counts().reset_index()
    stage_counts.columns = ['Stage', 'Count']
    fig_stage = px.bar(stage_counts, x='Stage', y='Count', title="Current Stage Distribution", color_discrete_sequence=['purple'])
    fig_stage.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Roboto", size=12, color="#1E3A8A"),
        xaxis_title="Current Stage",
        yaxis_title="Number of Customers"
    )
    st.plotly_chart(fig_stage, use_container_width=True)

    
    st.markdown("<h3>Cleaned Data Table</h3>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)


def visualize_phrasebank_data(df):
    if df is None or df.empty:
        st.error("No data available for Financial Phrasebank.")
        return
    st.markdown("<h2>Financial Phrasebank Sentiment Analysis</h2>", unsafe_allow_html=True)
    

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Positive Statements</h3>", unsafe_allow_html=True)
        positive_count = len(df[df['Sentiment'] == 'positive'])
        st.markdown(f"<p>{positive_count}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Neutral Statements</h3>", unsafe_allow_html=True)
        neutral_count = len(df[df['Sentiment'] == 'neutral'])
        st.markdown(f"<p>{neutral_count}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Negative Statements</h3>", unsafe_allow_html=True)
        negative_count = len(df[df['Sentiment'] == 'negative'])
        st.markdown(f"<p>{negative_count}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Sentiment Distribution
    st.markdown("<h3>Sentiment Distribution</h3>", unsafe_allow_html=True)
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig_sentiment = px.pie(sentiment_counts, names='Sentiment', values='Count', title="Sentiment Distribution", 
                           color_discrete_sequence=['green', 'red', 'grey'])
    fig_sentiment.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Roboto", size=12, color="#1E3A8A")
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

    
    st.markdown("<h3>Financial Phrasebank Data</h3>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

def visualize_stock_comparison():
    st.markdown("<h2>Stock Price Comparison</h2>", unsafe_allow_html=True)
    comparison_df = pd.DataFrame()
    for company in companies:
        df = data[company].copy()
        df['Company'] = company
        comparison_df = pd.concat([comparison_df, df[['Date', 'Close', 'Company']]], ignore_index=True)
    fig_comparison = px.line(comparison_df, x='Date', y='Close', color='Company', title="Stock Price Comparison")
    fig_comparison.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Roboto", size=12, color="#1E3A8A"),
        xaxis_title="Date",
        yaxis_title="Closing Price (USD)",
        legend_title="Company"
    )
    st.plotly_chart(fig_comparison, use_container_width=True)


def query_interface():
    st.markdown("<h2>Query Financial Data</h2>", unsafe_allow_html=True)
    query = st.text_input("Enter your query (e.g., $.Microsoft[?(@.Date == '2024-06-14')].Close)", 
                          placeholder="Enter JSONPath query or natural language question")
    if st.button("Run Query"):
        if query and router_engine:
            with st.spinner("Processing query..."):
                try:
                    response = router_engine.query(query)
                    st.success(f"Query Result: {response}")
                except Exception as e:
                    st.error(f"Error processing query: {e}")
        else:
            st.warning("Please enter a query or ensure query engine is available.")


def dashboard_page():
    with st.sidebar:
        st.markdown("<h2><i class='fas fa-chart-line'></i> Dashboard</h2>", unsafe_allow_html=True)
        st.markdown(f"<p><i class='fas fa-user'></i> Welcome, {st.session_state.username}</p>", unsafe_allow_html=True)
        analysis_type = st.selectbox("Select Analysis Type", 
                                    ["Stock Analysis", "Cleaned Data", "Financial Phrasebank", "Stock Comparison", "Query Interface"],
                                    help="Choose an analysis type")
        if analysis_type == "Stock Analysis":
            company = st.selectbox("Select Company", companies, help="Choose a company")
        if st.button("Log Out", key="logout_button", type="secondary"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()

    st.markdown("<h1 class='main-title'><i class='fas fa-chart-line'></i> Financial Insights Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<img src='https://via.placeholder.com/100?text=Logo' class='logo' alt='Project Logo'/>", unsafe_allow_html=True)
    st.markdown("<div class='company-data'>", unsafe_allow_html=True)

    if analysis_type == "Stock Analysis":
        st.markdown(f"<h2>{company} Financial Analysis</h2>", unsafe_allow_html=True)
        df = data.get(company)
        if df is None or df.empty:
            st.error(f"No data available for {company}.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Last Closing Price</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>${df['Close'].iloc[-1]:.2f}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Daily Change</h3>", unsafe_allow_html=True)
            change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            st.markdown(f"<p class='{'positive' if change >= 0 else 'negative'}'>{change:.2f}%</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Trading Volume</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>{df['Volume'].iloc[-1]:,}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<h3>Select Time Range</h3>", unsafe_allow_html=True)
        time_range = st.slider("Date Range", 
                               min_value=df['Date'].min().date(), 
                               max_value=df['Date'].max().date(), 
                               value=(df['Date'].max() - timedelta(days=30)).date(), 
                               format="YYYY-MM-DD")
        filtered_df = df[df['Date'].dt.date >= time_range]

        st.markdown("<h3>Price Trend</h3>", unsafe_allow_html=True)
        fig_price = px.line(filtered_df, x='Date', y='Close', title=f"{company} Stock Price")
        fig_price.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Roboto", size=12, color="#1E3A8A"),
            xaxis_title="Date",
            yaxis_title="Price (USD)"
        )
        st.plotly_chart(fig_price, use_container_width=True)

        st.markdown("<h3>Trading Volume</h3>", unsafe_allow_html=True)
        fig_volume = px.bar(filtered_df, x='Date', y='Volume', title=f"{company} Trading Volume")
        fig_volume.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Roboto", size=12, color="#1E3A8A"),
            xaxis_title="Date",
            yaxis_title="Volume"
        )
        st.plotly_chart(fig_volume, use_container_width=True)

        st.markdown("<h3>Historical Data</h3>", unsafe_allow_html=True)
        st.dataframe(filtered_df, use_container_width=True)

    elif analysis_type == "Cleaned Data":
        visualize_cleaned_data(data['cleaned'])
    elif analysis_type == "Financial Phrasebank":
        visualize_phrasebank_data(data['financial_phrasebank'])
    elif analysis_type == "Stock Comparison":
        visualize_stock_comparison()
    elif analysis_type == "Query Interface":
        query_interface()

    st.markdown("</div>", unsafe_allow_html=True)

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["Sign Up", "Log In"])
    with tab1:
        signup_page()
    with tab2:
        login_page()
else:
    dashboard_page()

