# Project Pulse - Jira Insights Generator
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://project-pulse-4pg63bggq3mwunnmcmagbl.streamlit.app/)
Transform raw Jira metrics into executive-ready insights using ChatGPT AI.

## 🎯 What This Demonstrates

**Product Management Skills:**
- Data → Strategy translation
- Executive communication
- Risk identification & mitigation
- Stakeholder management

**Technical Skills:**
- AI/LLM integration (OpenAI ChatGPT)
- RAG architecture (LangChain + ChromaDB)
- Vector embeddings & semantic search
- Data visualization (Plotly)
- Python development
- Production-ready code

## 🚀 Quick Start (Windows)

### 1. Setup Environment

```bash
# Create project folder
mkdir project-pulse
cd project-pulse

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Mock Data

```bash
python generate_mock_data.py
```

This creates `mock_jira_export.csv` with realistic data including:
- 6 sprints of issue history (Sprint 20-25)
- Bug spike in Sprint 23 (payment gateway integration)
- Tech debt accumulation pattern
- Critical bugs with extended cycle times
- Integration sprint bottlenecks

### 3. Get Google Gemini API Key (100% FREE!)

1. Go to [ai.google.dev](https://ai.google.dev)
2. Click **"Get API Key"**
3. Create in new or existing Google Cloud project
4. Copy the API key
5. **No credit card required!** Completely free tier

### 4. Run the App

```bash
streamlit run app.py
```

Browser opens automatically to `http://localhost:8501`

### 5. Use the App

1. **Enter API key** in the sidebar
2. **Upload** `mock_jira_export.csv`
3. **Explore insights:**
   - What changed since last month?
   - What should leadership worry about?
   - What's not obvious here?
4. **Generate** executive email updates

## 📁 Project Structure

```
project-pulse/
├── app.py                      # Main Streamlit application
├── generate_mock_data.py       # Creates realistic Jira CSV
├── mock_jira_export.csv        # Sample data (generated)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── SETUP_GUIDE.md             # Detailed Windows setup
```

## 🎨 Features

### 🎛️ Two Modes: Simple & RAG

**Simple Mode** (Default)
- Upload CSV → Get instant AI analysis
- Perfect for single sprint analysis
- Fast, no setup required

**RAG Mode** (Advanced)
- Enable RAG checkbox in sidebar
- Save sprint data to knowledge base (ChromaDB)
- AI retrieves similar past situations
- Get insights like: *"This pattern matches Sprint 23 from Project Phoenix where..."*

### 📊 Automated Analysis
- **Executive Summary**: Traffic light status (🟢🟡🔴) with key metrics
- **Trend Visualization**: Sprint velocity, bug rates, cycle times
- **Risk Detection**: Quality issues, tech debt, capacity problems

### 🤖 AI-Powered Insights (ChatGPT)
Ask questions like:
- "Why is velocity declining in recent sprints?"
- "What risks should we communicate to leadership?"
- "What patterns indicate future problems?"
- "How does our tech debt compare to industry standards?"

### 📧 Ready-to-Use Outputs
- **Email Generator**: Draft executive updates automatically
- **Interactive Charts**: Plotly visualizations you can explore
- **Decision Frameworks**: Risk severity + specific recommendations

## 🧠 What Makes This Special

### 1. Hybrid Architecture: Simple + RAG

**Smart Design Decision:**
- Most Jira analyses don't need RAG (single CSV fits in context)
- RAG adds value when comparing across multiple projects
- Built both to show you understand *when* to use RAG

**Simple Mode:**
```
Upload CSV → ChatGPT analyzes → Get insights
```

**RAG Mode:**
```
Upload CSV → Save to ChromaDB → Future analyses retrieve similar contexts
→ ChatGPT gets: Current data + Relevant past patterns → Better insights
```

**Why this impresses:**
- Shows architectural thinking (not every problem needs RAG)
- Demonstrates real RAG implementation (not just LLM + prompt)
- Modular design (RAG is additive, not required)

### 2. Solves a Real PM Pain Point
Every product manager struggles with:
- Drowning in Jira ticket noise
- Translating metrics into narratives
- Identifying what's actually important
- Communicating risks to executives

This tool does all of that automatically.

### 2. Realistic Mock Data with Hidden Insights
The generated data includes subtle patterns:
- **Bug Spike Pattern**: Sprint 23 shows payment gateway rush job
- **Tech Debt Accumulation**: 3 deferred refactors blocking features
- **Cycle Time Degradation**: Critical bugs taking 2x longer
- **Integration Tax**: Blocked tickets every 3rd sprint
- **Friday Deploy Curse**: 75% of bugs deployed end-of-week

### 3. Production-Quality RAG Implementation
- Clean, documented code
- Error handling for API failures
- Vector store with ChromaDB
- Semantic search using OpenAI embeddings
- LangChain for RAG orchestration
- Responsive UI with custom styling
- Works with real Jira exports
- Deployable to cloud in minutes

**RAG Stack:**
- **Vector DB:** ChromaDB (local, no cloud required)
- **Embeddings:** OpenAI text-embedding-3-small
- **Orchestration:** LangChain
- **Storage:** Persistent ChromaDB on disk

## 💰 Cost Breakdown

| Item | Cost |
|------|------|
| Python & Development | Free |
| Streamlit Hosting | Free (Community Cloud) |
| **Google Gemini API** | **100% FREE** |
| ChromaDB Storage | Free (local) |
| **Total** | **$0 - Completely Free!** |

**Why Gemini?**
- ✅ No credit card required
- ✅ 60 requests/minute free tier
- ✅ Gemini 1.5 Flash - fast & high quality
- ✅ Perfect for portfolio demos

## 🎓 Learning Outcomes

**For Interviewers/Reviewers:**
This project demonstrates:
- ✅ Product thinking (identifies real PM pain point)
- ✅ Technical execution (working AI + RAG integration)
- ✅ Architectural judgment (knows when RAG adds value)
- ✅ Communication skills (executive-ready outputs)
- ✅ Attention to detail (realistic data, polished UX)
- ✅ Full-stack capability (data → embeddings → vector DB → LLM → UI)

**For Users:**
Learn how to:
- Integrate OpenAI ChatGPT into applications
- Build AI-powered analytics tools
- Create executive dashboards with Streamlit
- Generate insights from structured data
- Deploy Python web apps

## 🔧 Customization Guide

### Understanding the RAG Architecture

**When Simple Mode is Enough:**
- Analyzing a single Jira export
- Data fits in LLM context window (~8K tokens)
- No need for historical comparison

**When RAG Mode Adds Value:**
- Comparing across multiple sprints/projects
- Learning from past similar situations
- Building institutional knowledge
- Questions like "How did we handle this before?"

**How RAG Works in This App:**

1. **Store Phase:**
   ```python
   Upload CSV → Generate summary → 
   Split into chunks → Create embeddings → 
   Store in ChromaDB with metadata
   ```

2. **Retrieve Phase:**
   ```python
   User asks question → Convert to embedding → 
   Semantic search in ChromaDB → 
   Retrieve top 3 similar past contexts
   ```

3. **Generate Phase:**
   ```python
   Combine: Current data + Retrieved context → 
   Send to ChatGPT → Get enhanced insights
   ```

**Example RAG Flow:**

*Without RAG:*
> "Bug rate increased 45% this sprint"

*With RAG:*
> "Bug rate increased 45% this sprint. This mirrors Project Phoenix Sprint 23 where rushed payment integration caused similar spike. Resolution: dedicated QA sprint + test coverage expansion"

### Use Real Jira Data

**Export from Jira:**
1. Navigate to your Jira project
2. Click **Issues** → **Search for issues**
3. Click **•••** → **Export** → **Export CSV (all fields)**
4. Upload to Project Pulse

### Modify AI Analysis Prompts

Edit `app.py` around line 35:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Your custom system prompt here..."},
        {"role": "user", "content": f"Your custom question: {question}"}
    ]
)
```

### Add Custom Visualizations

Use Plotly to add charts:

```python
import plotly.express as px

def create_priority_breakdown(df):
    fig = px.pie(df, names='Priority', title='Issues by Priority')
    return fig

# Then in main app:
st.plotly_chart(create_priority_breakdown(df))
```

### Change GPT Model

In `app.py`, modify the model parameter:

```python
model="gpt-4o"           # Best quality, higher cost
model="gpt-4o-mini"      # Faster, lower cost
model="gpt-3.5-turbo"    # Cheapest option
```

## 📊 Sample Questions to Ask the AI

**Strategic Level:**
- "What should our Q1 priorities be based on this data?"
- "Is our current delivery pace sustainable?"
- "What trade-offs should we communicate to leadership?"

**Tactical Level:**
- "Why is the backend becoming a bottleneck?"
- "Which bugs indicate systemic quality issues?"
- "Are we over-committing in sprint planning?"

**Executive Level:**
- "Summarize this for the CEO in 3 bullet points"
- "What's the one thing leadership must know?"
- "Draft an all-hands project update"

## 🚀 Deployment to Streamlit Cloud

### Step 1: Push to GitHub

```bash
# Install Git for Windows from git-scm.com

git init
git add .
git commit -m "Initial commit: Project Pulse"

# Create repo on GitHub.com, then:
git remote add origin https://github.com/yourusername/project-pulse.git
git push -u origin main
```

### Step 2: Deploy to Streamlit

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Connect your GitHub account
4. Select repository: `project-pulse`
5. Main file path: `app.py`
6. Click **Deploy**

Your app will be live at: `https://yourapp.streamlit.app`

### Step 3: Set API Key (Optional)

In Streamlit Cloud dashboard:
1. **App settings** → **Secrets**
2. Add:
```toml
OPENAI_API_KEY = "sk-proj-your-key-here"
```

**Portfolio Tip:** For demos, let viewers use their own key to avoid sharing costs.

## 💡 Extension Ideas

1. **Enhanced RAG Features**:
   - Multi-modal search (search by metrics, not just text)
   - Time-weighted relevance (recent sprints ranked higher)
   - Project clustering (group similar projects automatically)

2. **Team Collaboration**:
   - Shared knowledge bases across team
   - Comment threads on insights
   - Bookmark important analyses

3. **Integrations**:
   - **Real-time Jira API**: Auto-sync instead of CSV upload
   - **Slack Bot**: Weekly insights posted automatically
   - **Email Digest**: Schedule executive summaries

4. **Advanced Analytics**:
   - Predictive: Forecast next sprint outcomes using historical data
   - Anomaly Detection: Auto-flag unusual patterns
   - Benchmarking: Compare against industry standards

5. **RAG Improvements**:
   - **Hybrid Search**: Combine semantic + keyword search
   - **Re-ranking**: Improve retrieval accuracy
   - **Source Attribution**: Show which past projects informed each insight

## 🎤 Demo Tips for Interviews

**Opening (30 seconds):**
> "I built Project Pulse to solve a problem every PM faces: drowning in Jira data but struggling to communicate insights to executives. It uses ChatGPT to transform raw metrics into narratives that leadership actually cares about."

**Demo Flow (2-3 minutes):**
1. **Simple Mode Demo:**
   - Upload CSV → "See instant AI analysis"
   - Click "Analyze Changes" → Show insights

2. **RAG Mode Demo:**
   - Enable RAG checkbox → "Now let's build knowledge"
   - Upload same CSV → Save as "Project Alpha"
   - Upload another CSV → Save as "Project Beta"
   - Ask: "How does this compare to past projects?"
   - Show RAG retrieving context → Enhanced insights

3. **Show the difference:**
   - "Without RAG: analyzes current data only"
   - "With RAG: learns from 2 past projects, provides context"

**Technical Deep-Dive (if asked):**
- "Uses OpenAI GPT-4 with contextual prompts"
- "Implements true RAG with ChromaDB vector store"
- "Semantic search using OpenAI embeddings"
- "LangChain for RAG orchestration"
- "Modular architecture - RAG is optional, not required"

**Business Value:**
- "Saves 2+ hours per week on status reporting"
- "Improves stakeholder communication quality"
- "Catches risks before they become crises"

**Questions You'll Get:**

*"How does this compare to Jira's built-in analytics?"*
→ "Jira shows what happened. This explains why it matters, what to do, and learns from past similar situations when RAG mode is enabled."

*"Would real PMs actually use this?"*
→ "Absolutely. I built it because I was tired of manually connecting dots between sprints. The RAG mode is especially useful for new PMs joining a team - instant access to project history."

*"Is this really RAG or just LLM with context?"*
→ "Great question! Simple mode is LLM + context injection. RAG mode implements true RAG: vector embeddings in ChromaDB, semantic similarity search, and context retrieval. I built both to show I understand when each approach is appropriate."

*"What if the AI hallucinates?"*
→ "In simple mode, it only analyzes provided data. In RAG mode, retrieval is based on actual stored summaries, not generated. But you're right - in production I'd add source citations and confidence scores."

*"How would you monetize this?"*
→ "Freemium: Free simple mode. Pro tier ($15/user/mo) includes RAG mode, unlimited knowledge base, Slack integration, and team collaboration features."

## 🐛 Troubleshooting

### Windows-Specific Issues

**Virtual environment won't activate:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**"Python is not recognized":**
- Reinstall Python from python.org
- Check "Add Python to PATH" during install
- Restart terminal

**Port 8501 in use:**
```bash
streamlit run app.py --server.port 8502
```

### OpenAI API Issues

**"Invalid API key":**
- Key should start with `sk-proj-` or `sk-`
- No spaces before/after when pasting
- Generate new key if needed

**"Rate limit exceeded":**
- Free credits exhausted
- Add payment method: platform.openai.com/account/billing
- Set monthly spending limit ($10 is plenty)

**"Quota exceeded":**
- Check usage: platform.openai.com/account/usage
- Upgrade plan or wait for reset

## 📚 Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **OpenAI API Docs**: https://platform.openai.com/docs
- **Plotly Documentation**: https://plotly.com/python/
- **Jira Export Guide**: https://support.atlassian.com/jira-software-cloud/docs/export-issues/

## 📝 License

MIT License - Free to use for portfolios, demos, and commercial projects.

## 🙋 Questions?

Built to demonstrate PM + AI skills for portfolio/interviews.

**Feel free to:**
- Clone and customize for your needs
- Use in job applications
- Extend with your own features
- Share with other PMs

---

**Tech Stack:** Python • Streamlit • Google Gemini • LangChain • ChromaDB • Plotly • Pandas

**API:** 100% Free with Google Gemini

**Time to Build:** Weekend project → Portfolio piece

**Impact:** Solves real PM pain point with measurable value