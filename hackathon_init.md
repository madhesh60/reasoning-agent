Your Complete Task List

TASK 1 — Get Azure Account (Do This Today)
Step 1: Go to azure.microsoft.com/free/students
Step 2: Click "Activate Now"
Step 3: Sign in with a personal Microsoft account (like outlook.com or hotmail). Don't use your college email to sign in — use it only to verify your student status.
Step 4: It will ask for your institution email (your CIT email ending in @citchennai.net or similar). Enter it to verify you're a student.
Step 5: Check your college email for a verification link and click it.
Step 6: Done. You now have $100 in free credits. Go to portal.azure.com — you should see your subscription.

TASK 2 — Set Up Azure AI Foundry (Your Agent Platform)
Step 1: Go to ai.azure.com
Step 2: Sign in with the same Microsoft account you used in Task 1.
Step 3: Click "Create a project"
Step 4: Give it a name like reasoning-agent-hack
Step 5: It will ask you to create a Hub (think of a hub like a folder that holds your project). Click "Create new hub", name it anything, pick region East US (this region has the best model availability).
Step 6: Click Create. Wait 2–3 minutes.
Step 7: Once your project opens, go to the left sidebar → "Models + Endpoints" → "Deploy Model"
Step 8: Search for phi-4-mini-reasoning or phi-4-reasoning or both — deploy this first. It's cheaper and fast.
Step 9: After deployment, copy the Endpoint URL and API Key shown. Save them in a .env file on your laptop. You'll need these in your code.

TASK 3 — Set Up Your Local Dev Environment
Step 1: Make sure Python 3.11+ is installed. Run python --version in your terminal.
Step 2: Create a project folder: mkdir reasoning-agent && cd reasoning-agent
Step 3: Create a virtual environment (a clean box just for this project): python -m venv venv then venv\Scripts\activate on Windows.
Step 4: Install the packages you need:
pip install langgraph langchain-openai azure-ai-projects python-dotenv fastapi uvicorn
Step 5: Create a .env file in your folder and put:
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_KEY=your_key_here
Step 6: Test it works — write a 5-line Python file that calls the model and prints a response. If it prints something, your setup is working.

TASK 4 — Build Your Multi-Agent System with LangGraph
This is the main coding work. You're building 4 agents that each do one job.
Step 1: Create a file called agents.py
Step 2: Build the Planner Agent — it takes the user's question and breaks it into 3–4 smaller sub-questions. Example: "EV market risks in India" → "What is the current EV policy?", "What are funding risks?", etc.
Step 3: Build the Researcher Agent — it takes one sub-question, searches the web or a document store, and returns raw facts. Use LangChain's built-in web search tool (Tavily Search — free tier available).
Step 4: Build the Analyst Agent — it reads the raw facts and pulls out only the important insights. Think of it like a student who highlights the key points in a textbook.
Step 5: Build the Writer Agent — it takes all insights and formats them into a clean, structured report with headings and bullet points.
Step 6: Create a file called graph.py — here you connect these 4 agents using LangGraph. Think of it like drawing arrows between the agents: Planner → Researcher → Analyst → Writer.
Step 7: Test the full chain by running it with a sample question and checking if the output makes sense.

TASK 5 — Connect A2A (Agents Talking to Each Other via Microsoft's Protocol)
This is what makes your project stand out.
Step 1: Go back to ai.azure.com → your project → Tools tab
Step 2: Click "Connect tool" → "Custom" → "Agent2Agent"
Step 3: For each of your sub-agents (Researcher, Analyst, Writer), you'll expose them as A2A endpoints — meaning each one gets its own web address that can be called by the orchestrator.
Step 4: In your code, update the Planner agent so instead of calling the other agents directly in Python, it calls them via their A2A endpoint URLs. This is like upgrading from agents talking inside one room to agents calling each other on the phone across different locations.
Step 5: Test it — send a question to the Planner and verify the chain runs through A2A.

TASK 6 — Add MCP Tools (Give Agents Superpowers)
Step 1: In ai.azure.com → your project → Tools tab → "Connect tool" → look for "Web Search" and enable it.
Step 2: Also enable "Azure AI Search" if you want to let agents search through documents.
Step 3: In your agent code, connect to the MCP toolbox endpoint (Foundry gives you a URL for this). Your Researcher Agent will now use Web Search through MCP instead of a standalone search library.
Step 4: Test by asking your agent a question about something recent (like "latest AI news") and verify it actually searches the web.

TASK 7 — Build a Simple API + Docker Container
Step 1: Create a file called main.py with a simple FastAPI web server. It needs one endpoint: POST /ask that accepts a question and returns the agent's report.
Step 2: Test it locally: uvicorn main:app --reload — open your browser to localhost:8000/docs and try the endpoint.
Step 3: Create a Dockerfile in your project folder:
dockerfileFROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
Step 4: Build the Docker image: docker build -t reasoning-agent .
Step 5: Run it locally to test: docker run -p 8000:8000 reasoning-agent — make sure it still works.

TASK 8 — Deploy to Azure Container Apps (Cloud Deployment)
Step 1: Install Azure CLI on your laptop: winget install Microsoft.AzureCLI
Step 2: Log in: az login — it opens a browser window, sign in with your Microsoft account.
Step 3: Run these 3 commands:
bashaz group create --name reasoning-agent-rg --location eastus

az acr create --name reasoningagentregistry --resource-group reasoning-agent-rg --sku Basic --admin-enabled true

az containerapp up --name reasoning-agent-app --resource-group reasoning-agent-rg --source .
Step 4: Azure gives you a public URL after deployment. Copy it. That's your live production URL that judges can visit.
Step 5: Test the live URL — send a request to it and make sure it responds correctly.

TASK 9 — Build a Simple UI (So the Demo Looks Clean)
Step 1: Create a simple HTML + JavaScript frontend. Just a text box for the question and a box that shows the agent's report.
Step 2: Wire it up to your Azure Container App URL.
Step 3: Or use Streamlit (even easier): pip install streamlit, create app.py, run streamlit run app.py. Streamlit gives you a working UI in 20 lines of Python.

TASK 10 — Architecture Diagram (Required for Submission)
Step 1: Go to draw.io (free, no sign up needed)
Step 2: Draw boxes for: User → Frontend → Azure Container Apps → Foundry Agent Service → LangGraph Orchestrator → A2A Sub-Agents → MCP Tools → Web Search / Azure AI Search
Step 3: Add arrows connecting them to show the flow.
Step 4: Export as PNG. This goes in your GitHub repo and your submission.

TASK 11 — GitHub Repo (Required for Submission)
Step 1: Create a new public repo on GitHub under your madhesh60 account. Name it reasoning-agent-hackathon.
Step 2: Write a clear README.md with: what problem it solves, how to run it, and a screenshot of the architecture diagram.
Step 3: Push all your code to it.
Step 4: Make sure anyone can clone it and run it. That's what judges will try to do.

TASK 12 — Demo Video (Required for Submission)
Step 1: Use OBS Studio (free) or Windows Game Bar (Win + G) to record your screen.
Step 2: Record yourself: type a complex question into your UI → show the agents thinking step by step in the terminal logs → show the final clean report output.
Step 3: Keep it under 5 minutes. The flow should be: problem intro (30 sec) → demo (3.5 min) → architecture walkthrough (1 min).
Step 4: Upload to YouTube as unlisted. Copy the link.

TASK 13 — Submit
Step 1: Go to the hackathon platform (link comes in your registration confirmation email)
Step 2: Click "Create Project"
Step 3: Fill in: project description, YouTube link, GitHub repo link, architecture diagram
Step 4: Submit before June 14, 11:59 PM Pacific Time (that's June 15, 12:29 PM IST for you in Salem)