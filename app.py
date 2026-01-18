import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import google.generativeai as genai
import os
import json

# RAG components
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Page config
st.set_page_config(
    page_title="Project Pulse - Jira Insights",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .insight-box {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .rag-box {
        background-color: #faf5ff;
        border-left: 4px solid #9333ea;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for RAG
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'knowledge_base_count' not in st.session_state:
    st.session_state.knowledge_base_count = 0

def initialize_vector_store(api_key):
    """Initialize ChromaDB vector store with local embeddings"""
    if st.session_state.vector_store is None:
        # Use local embeddings (no API calls!)
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Small, fast, free!
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        st.session_state.vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
    return st.session_state.vector_store

def add_to_knowledge_base(df, summary_text, project_name, api_key):
    """Add sprint data to RAG knowledge base"""
    vector_store = initialize_vector_store(api_key)
    
    # Create document from sprint summary
    doc_text = f"""
    Project: {project_name}
    Date: {datetime.now().strftime('%Y-%m-%d')}
    
    Sprint Summary:
    - Total Issues: {len(df)}
    - Bug Rate: {len(df[df['Issue Type'] == 'Bug']) / len(df) * 100:.1f}%
    - Completion Rate: {len(df[df['Status'] == 'Done']) / len(df) * 100:.1f}%
    - Average Velocity: {df.groupby('Sprint')['Story Points'].sum().mean():.0f}
    
    Key Findings:
    {summary_text}
    
    Issue Distribution:
    {df['Issue Type'].value_counts().to_dict()}
    
    Sprint Breakdown:
    {df.groupby('Sprint').agg({'Story Points': 'sum', 'Issue Key': 'count'}).to_string()}
    """
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = text_splitter.split_text(doc_text)
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "project": project_name,
                "date": datetime.now().isoformat(),
                "type": "sprint_summary"
            }
        )
        for chunk in chunks
    ]
    
    # Add to vector store
    vector_store.add_documents(documents)
    st.session_state.knowledge_base_count += 1
    
    return True

def retrieve_similar_context(query, api_key, k=3):
    """Retrieve similar past sprints from knowledge base"""
    if st.session_state.vector_store is None:
        return None
    
    try:
        docs = st.session_state.vector_store.similarity_search(query, k=k)
        if not docs:
            return None
        
        context = "\n\n---\n\n".join([
            f"Past Project Context:\n{doc.page_content}" 
            for doc in docs
        ])
        return context
    except:
        return None

def get_available_model(api_key):
    """Get the first available Gemini model"""
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        
        # Prefer gemini-2.5-flash (newest and fastest)
        for m in models:
            if 'gemini-2.5-flash' in m.name and 'generateContent' in m.supported_generation_methods:
                return m.name
        
        # Fallback to gemini-2.0-flash
        for m in models:
            if 'gemini-2.0-flash' in m.name and 'generateContent' in m.supported_generation_methods:
                return m.name
        
        # Fallback to any gemini-1.5
        for m in models:
            if 'gemini-1.5' in m.name and 'generateContent' in m.supported_generation_methods:
                return m.name
        
        # Use first available model with generateContent
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                return m.name
                
        return None
    except Exception as e:
        return None

def analyze_with_gemini(df, question, api_key, use_rag=False):
    """Send data to Gemini for analysis with optional RAG"""
    
    # Get available model
    model_name = get_available_model(api_key)
    
    if not model_name:
        return "Error: Could not find available Gemini model. Please check your API key at ai.google.dev"
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # Prepare data summary
    data_summary = f"""
    Current Jira Metrics:
    - Total Issues: {len(df)}
    - Date Range: {df['Created'].min()} to {df['Created'].max()}
    - Issue Types: {df['Issue Type'].value_counts().to_dict()}
    - Status Distribution: {df['Status'].value_counts().to_dict()}
    - Priority Distribution: {df['Priority'].value_counts().to_dict()}
    
    Recent Trends:
    {df.groupby('Sprint')['Issue Key'].count().tail(5).to_string()}
    
    Key Metrics by Sprint:
    {df.groupby('Sprint').agg({'Story Points': 'sum', 'Issue Key': 'count'}).to_string()}
    
    Bug Trend:
    {df[df['Issue Type'] == 'Bug'].groupby('Sprint').size().to_string()}
    """
    
    # Retrieve context from knowledge base if RAG enabled
    rag_context = ""
    if use_rag and st.session_state.knowledge_base_count > 0:
        retrieved = retrieve_similar_context(question, api_key)
        if retrieved:
            rag_context = f"\n\nHistorical Context from Similar Past Projects:\n{retrieved}\n"
    
    prompt = f"""You are a senior product manager with expertise in analyzing Jira metrics and communicating insights to executives.

When historical context is provided, use it to:
1. Compare current metrics to past similar situations
2. Reference what worked/didn't work before
3. Provide more informed recommendations based on patterns

Analyze this Jira data:

{data_summary}
{rag_context}

Question: {question}

Provide a concise, executive-level response that:
1. Identifies key patterns and trends
2. Highlights risks and opportunities  
3. Provides actionable recommendations
4. Uses specific numbers from the data
{"5. References relevant patterns from historical context when applicable" if rag_context else ""}

Format your response in clear sections with bullet points."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}\n\nPlease check your API key and try again."

def create_velocity_chart(df):
    """Create sprint velocity trend chart"""
    velocity_data = df.groupby('Sprint').agg({
        'Story Points': 'sum',
        'Issue Key': 'count'
    }).reset_index()
    velocity_data.columns = ['Sprint', 'Story Points', 'Issue Count']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=velocity_data['Sprint'],
        y=velocity_data['Story Points'],
        name='Story Points',
        marker_color='#3b82f6'
    ))
    
    fig.update_layout(
        title='Sprint Velocity Trend',
        xaxis_title='Sprint',
        yaxis_title='Story Points',
        height=400
    )
    
    return fig

def create_bug_trend_chart(df):
    """Create bug trend analysis"""
    bug_data = df[df['Issue Type'] == 'Bug'].groupby('Sprint').size().reset_index(name='Bug Count')
    
    fig = px.line(bug_data, x='Sprint', y='Bug Count', 
                  title='Bug Trend Over Time',
                  markers=True)
    fig.update_traces(line_color='#ef4444', marker=dict(size=10))
    
    return fig

def create_cycle_time_chart(df):
    """Create cycle time distribution"""
    df_resolved = df[df['Status'] == 'Done'].copy()
    df_resolved['Cycle Time (days)'] = (
        pd.to_datetime(df_resolved['Resolved']) - pd.to_datetime(df_resolved['Created'])
    ).dt.days
    
    fig = px.box(df_resolved, x='Issue Type', y='Cycle Time (days)',
                 title='Cycle Time by Issue Type',
                 color='Issue Type')
    
    return fig

def generate_executive_summary(df):
    """Generate automated executive summary"""
    total_issues = len(df)
    bugs = len(df[df['Issue Type'] == 'Bug'])
    bug_rate = (bugs / total_issues) * 100
    
    done_issues = len(df[df['Status'] == 'Done'])
    completion_rate = (done_issues / total_issues) * 100
    
    avg_velocity = df.groupby('Sprint')['Story Points'].sum().mean()
    
    # Determine status
    if bug_rate > 30 or completion_rate < 60:
        status = "🔴 ATTENTION NEEDED"
        status_class = "warning-box"
    elif bug_rate > 20 or completion_rate < 75:
        status = "🟡 MONITOR CLOSELY"
        status_class = "insight-box"
    else:
        status = "🟢 ON TRACK"
        status_class = "success-box"
    
    return {
        'status': status,
        'status_class': status_class,
        'total_issues': total_issues,
        'bug_rate': bug_rate,
        'completion_rate': completion_rate,
        'avg_velocity': avg_velocity
    }

# Main App
st.title("📊 Project Pulse")
st.subheader("Transform Jira metrics into executive-ready insights with AI + RAG")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input("Google Gemini API Key", type="password", 
                            help="Get your FREE API key from ai.google.dev")
    
    # Debug: Show available models
    if api_key:
        with st.expander("🔍 Debug: Check API Key", expanded=False):
            if st.button("Test API Key"):
                try:
                    genai.configure(api_key=api_key)
                    models = list(genai.list_models())
                    st.success(f"✅ API Key valid! Found {len(models)} models")
                    
                    available = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
                    if available:
                        st.info(f"Available models: {', '.join(available[:3])}")
                    else:
                        st.warning("No models found with generateContent support")
                except Exception as e:
                    st.error(f"❌ API Key error: {str(e)}")
                    st.info("Get a new key at: https://ai.google.dev")
    
    st.markdown("---")
    
    # RAG Mode Toggle
    st.subheader("🧠 RAG Knowledge Base")
    use_rag = st.checkbox(
        "Enable RAG Mode",
        help="Use historical project data to enhance insights"
    )
    
    if st.session_state.knowledge_base_count > 0:
        st.success(f"✅ {st.session_state.knowledge_base_count} projects in knowledge base")
    else:
        st.info("📚 No historical data yet")
    
    if st.button("🗑️ Clear Knowledge Base"):
        if os.path.exists("./chroma_db"):
            import shutil
            shutil.rmtree("./chroma_db")
        st.session_state.vector_store = None
        st.session_state.knowledge_base_count = 0
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### 🚀 How to use:
    
    **Simple Mode:**
    1. Upload Jira CSV
    2. Get instant AI insights
    
    **RAG Mode (Advanced):**
    1. Enable RAG checkbox ☑️
    2. Upload & save to knowledge base
    3. Get insights informed by past projects
    
    ### 💡 Sample Questions:
    - What changed since last month?
    - What should leadership worry about?
    - How does this compare to past sprints?
    - What patterns match previous issues?
    
    ### 🆓 Get FREE API Key:
    1. Visit: [ai.google.dev](https://ai.google.dev)
    2. Click "Get API Key"
    3. Create in new/existing project
    4. Copy & paste here!
    
    **Free Tier:** 60 requests/min
    
    **Note:** RAG uses local embeddings (no API!)
    - Embeddings run on your computer
    - Only Gemini text generation uses API
    - No embedding quota limits!
    """)

# File Upload
uploaded_file = st.file_uploader("Upload Jira Export (CSV)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Display data overview
    st.success(f"✅ Loaded {len(df)} issues from Jira")
    
    # RAG: Save to Knowledge Base option
    if api_key and use_rag:
        with st.expander("💾 Save to Knowledge Base", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                project_name = st.text_input(
                    "Project Name",
                    value=f"Project-{datetime.now().strftime('%Y%m%d')}",
                    help="Give this sprint data a memorable name"
                )
            with col2:
                st.write("")
                st.write("")
                if st.button("💾 Save", type="primary"):
                    with st.spinner("Adding to knowledge base..."):
                        summary = generate_executive_summary(df)
                        summary_text = f"Status: {summary['status']}, Bug Rate: {summary['bug_rate']:.1f}%, Completion: {summary['completion_rate']:.1f}%"
                        
                        success = add_to_knowledge_base(df, summary_text, project_name, api_key)
                        if success:
                            st.success(f"✅ Added to knowledge base!")
                            st.rerun()
    
    with st.expander("📋 View Raw Data"):
        st.dataframe(df.head(20))
    
    # Executive Summary
    st.markdown("---")
    st.header("🎯 Executive Summary")
    
    summary = generate_executive_summary(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Issues", summary['total_issues'])
    with col2:
        st.metric("Bug Rate", f"{summary['bug_rate']:.1f}%")
    with col3:
        st.metric("Completion Rate", f"{summary['completion_rate']:.1f}%")
    with col4:
        st.metric("Avg Velocity", f"{summary['avg_velocity']:.0f} pts")
    
    st.markdown(f"""
    <div class="{summary['status_class']}">
        <h3>{summary['status']}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # RAG Status Indicator
    if use_rag and st.session_state.knowledge_base_count > 0:
        st.markdown(f"""
        <div class="rag-box">
            <strong>🧠 RAG Mode Active:</strong> Analysis enhanced with insights from {st.session_state.knowledge_base_count} past project(s)
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("---")
    st.header("📈 Key Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_velocity_chart(df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_bug_trend_chart(df), use_container_width=True)
    
    st.plotly_chart(create_cycle_time_chart(df), use_container_width=True)
    
    # AI-Powered Insights
    st.markdown("---")
    st.header("🤖 AI-Powered Insights" + (" + RAG" if use_rag else "") + " (Gemini)")
    
    if api_key:
        tab1, tab2, tab3, tab4 = st.tabs([
            "What Changed?", 
            "Leadership Concerns", 
            "Hidden Insights",
            "Custom Question"
        ])
        
        with tab1:
            if st.button("Analyze Changes", key="changes"):
                with st.spinner("Gemini analyzing trends..."):
                    try:
                        analysis = analyze_with_gemini(df, 
                            "What are the most significant changes in metrics compared to previous sprints? Focus on velocity, quality, and delivery predictability.",
                            api_key, use_rag)
                        st.markdown(analysis)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("💡 Make sure your Gemini API key is valid. Get one FREE at ai.google.dev")
        
        with tab2:
            if st.button("Identify Risks", key="risks"):
                with st.spinner("Gemini identifying risks..."):
                    try:
                        analysis = analyze_with_gemini(df,
                            "What should executive leadership be most concerned about? Identify top 3 risks with business impact and recommendations.",
                            api_key, use_rag)
                        st.markdown(analysis)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with tab3:
            if st.button("Find Hidden Patterns", key="patterns"):
                with st.spinner("Gemini finding patterns..."):
                    try:
                        analysis = analyze_with_gemini(df,
                            "What patterns or insights are not immediately obvious from basic metrics? Look for correlations, timing patterns, or systemic issues.",
                            api_key, use_rag)
                        st.markdown(analysis)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with tab4:
            custom_question = st.text_area(
                "Ask a custom question about the data:",
                placeholder="e.g., How does this sprint compare to past similar situations?"
            )
            if st.button("Get Answer", key="custom"):
                if custom_question:
                    with st.spinner("Gemini analyzing..."):
                        try:
                            analysis = analyze_with_gemini(df, custom_question, api_key, use_rag)
                            st.markdown(analysis)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    else:
        st.warning("⚠️ Add your Gemini API key in the sidebar to unlock AI insights")
        st.info("Get your FREE API key from: https://ai.google.dev")
    
    # Email Generator
    st.markdown("---")
    st.header("📧 Generate Executive Email")
    
    if st.button("Generate Email Draft"):
        if api_key:
            with st.spinner("Gemini crafting email..."):
                try:
                    email = analyze_with_gemini(df,
                        """Write a concise executive email update (200-300 words) that includes:
                        1. Brief status overview with traffic light indicator
                        2. Top 3 wins/positives
                        3. Top 2-3 concerns with specific metrics
                        4. Clear recommendations
                        
                        Use a professional but approachable tone. Include specific metrics. 
                        Start with 'Subject:' line.""",
                        api_key, use_rag)
                    
                    st.markdown("### 📨 Draft Email:")
                    st.info(email)
                    st.download_button(
                        label="📥 Download Email",
                        data=email,
                        file_name="executive_update.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("⚠️ API key required")

else:
    st.info("👆 Upload a Jira CSV file to get started")
    
    # Show instructions
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 🎯 Simple Mode
        
        **Perfect for:** Single sprint analysis
        
        1. Upload Jira CSV
        2. Get instant AI insights
        3. No setup needed
        
        **Use when:** You need quick analysis of current sprint
        """)
    
    with col2:
        st.markdown("""
        ## 🧠 RAG Mode (Advanced)
        
        **Perfect for:** Multi-project learning
        
        1. Enable RAG checkbox
        2. Upload multiple sprint CSVs
        3. Save each to knowledge base
        4. Get insights informed by history
        
        **Use when:** You want to learn from past patterns
        """)
    
    st.markdown("---")
    st.markdown("""
    ## 🆓 Google Gemini API - Completely FREE!
    
    ### Why Gemini?
    - ✅ **Generous Free Tier**: 60 requests per minute
    - ✅ **No Credit Card Required**: Unlike OpenAI
    - ✅ **High Quality**: Gemini 1.5 Flash is fast and smart
    - ✅ **Perfect for Demos**: Unlimited for portfolio projects
    
    ### Get Your FREE API Key:
    1. Visit: **[ai.google.dev](https://ai.google.dev)**
    2. Click **"Get API Key"**
    3. Create in new or existing Google Cloud project
    4. Copy the key
    5. Paste in sidebar 👈
    
    **That's it!** No payment info, no trials - just free AI.
    
    ---
    
    ## 📖 How RAG Works Here
    
    **Without RAG (Simple Mode):**
    - AI analyzes only the current CSV
    - No historical context
    - Fast, straightforward insights
    
    **With RAG (Advanced Mode):**
    - Stores past sprint summaries in vector database (ChromaDB)
    - Retrieves similar historical situations
    - AI answers like: *"This bug spike pattern is similar to Sprint 23 from Project Phoenix, where the root cause was rushed integration work"*
    - Learns from your project history
    
    ### 🎓 Why This Architecture?
    
    - **Modular:** RAG is optional, not required
    - **Scalable:** Add unlimited historical projects
    - **Smart:** Semantic search finds relevant past contexts
    - **Practical:** Shows real understanding of when RAG adds value
    """)
    
    st.success("💡 **Tip:** Try simple mode first, then enable RAG after uploading 2-3 different sprint exports to see the difference!")