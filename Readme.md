# Real Estate AI Calling Agent

## 1. Project Overview

This project is a sophisticated, voice-based AI calling agent designed for a real estate developer. It is capable of handling customer inquiries over the phone, providing factually accurate information, and adapting its language (English/Hinglish) based on the caller.

The agent leverages a powerful Retrieval-Augmented Generation (RAG) architecture to answer questions based on a dynamic knowledge base of real estate data, ensuring its responses are grounded in real-time information rather than static training.

**Core Features:**

* **Voice-Based Interaction:** Handles live phone calls, transcribing user speech and generating spoken responses.
* **Fact-Based Answers:** Utilizes a RAG system connected to a live database (NeonDB) to provide accurate details on projects, availability, and sales.
* **Intelligent Query Planning:** Employs a two-step LLM process (Planner/Answerer) to understand user intent, formulate optimal database queries, and generate context-aware responses.
* **Bilingual Support:** Automatically adapts its response language (English/Hinglish) based on the user's query.
* **Secure and Scalable:** Designed for deployment in a containerized environment (Docker) with secure, token-based authentication.

## 2. System Architecture

The agent operates on a sophisticated, multi-stage pipeline that processes audio from a live phone call, reasons about the user's query, retrieves relevant data, and generates a spoken response in near real-time.

The workflow is as follows:

1.  **Telephony:** Exotel receives an incoming call and establishes a WebSocket connection, streaming the user's voice audio to the application.
2.  **Speech-to-Text (STT):** The application receives the audio and uses the OpenAI Whisper API to transcribe it into text.
3.  **Planner LLM:** The transcribed text is sent to a fast "Planner" LLM (gpt-4o-mini) which analyzes the query, translates it into an optimal search query, and identifies the correct data table to filter by.
4.  **RAG Retrieval:** The application uses the plan from the Planner to query the vector store (NeonDB), retrieving the most relevant factual documents.
5.  **Answerer LLM:** The original query, the retrieved context, and a system prompt are all sent to the main "Answerer" LLM (gpt-4o), which generates the final, conversational response.
6.  **Text-to-Speech (TTS):** The text response is sent to the ElevenLabs API, which converts it into a high-quality, natural-sounding voice.
7.  **Playback:** The generated audio is streamed back through the WebSocket to Exotel, which plays it to the user on the call.

## 3. Technology Stack

| Component         | Service / Library               | Purpose                                                                |
| ----------------- | ------------------------------- | ---------------------------------------------------------------------- |
| **Web Framework** | FastAPI                         | High-performance Python framework for the API and WebSocket server.    |
| **Web Server** | Uvicorn / Gunicorn              | Runs the FastAPI application in production.                            |
| **Telephony** | Exotel                          | Provides the virtual phone number and handles the live call stream.    |
| **Speech-to-Text** | OpenAI Whisper                  | Transcribes user's spoken audio into text.                             |
| **Core AI Logic** | OpenAI GPT-4o / GPT-4o-mini     | Powers the two-step "Planner" and "Answerer" reasoning engine.         |
| **Text-to-Speech** | ElevenLabs                      | Converts the AI's text response into a natural human voice.            |
| **Vector Database** | NeonDB with pgvector            | Stores and serves the factual knowledge base for the RAG system.       |
| **Embedding Model** | Hugging Face (instructor-large) | Creates the vector embeddings for the RAG knowledge base.              |
| **Orchestration** | LangChain                       | Simplifies interaction with the vector database (langchain-postgres).  |
| **Deployment** | Docker & Hugging Face Spaces    | Containerizes the application for a stable, secure, and permanent deployment. |

## 4. Project Workflow & Setup

This project is divided into two main workflows: the **Data Pipeline** (a one-time setup to build the AI's knowledge) and the **Live Agent** (the deployed application).

### Phase 1: The Data Pipeline (Building the Knowledge Base)

This two-step process populates your NeonDB vector store.

**Step 1: Upload Raw Data to Database**

* **Purpose:** To get your 6 source CSV files into standard SQL tables in NeonDB.
* **Script:** `upload_csvs_to_db.py`
* **Action:**
    1.  Ensure your source CSV files are in the correct directories as defined in the script.
    2.  Run the script from your terminal: `python upload_csvs_to_db.py`
* **Outcome:** 6 new tables (e.g., `ongoing_projects_source`) are created in your NeonDB, populated with your raw data.

**Step 2: Create the Vector Store**

* **Purpose:** To read the raw data from the tables, create embeddings, and build the final, searchable RAG knowledge base.
* **Script:** `ingest_rag_data.py`
* **Action:**
    1.  Ensure the `SOURCE_TABLE_NAMES` list in the script matches the tables created in Step 1.
    2.  Run the script from your terminal: `python ingest_rag_data.py`
* **Outcome:** The `real_estate_embeddings` collection is created in your NeonDB, containing the vector embeddings for your entire knowledge base.

### Phase 2: The Live Agent

**Local Development & Testing**

* **Environment Setup:**
    1.  Create a `.env` file in the project root.
    2.  Add your secret keys: `NEON_DATABASE_URL`, `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, `SHARED_SECRET`.
* **Run Locally with Docker:**
    1.  Run `docker compose up --build -d` from the terminal.
    2.  Your application will be running and accessible at `http://localhost:8000`.
* **Testing:**
    1.  Go to `http://localhost:8000/docs` to access the Swagger UI. Use the `/test-text-query` endpoint to test the RAG and LLM logic with text input.
    2.  Use the `websocket_test_client.py` script (`python websocket_test_client.py`) to simulate an audio call to the local Docker container.

**Deployment to Hugging Face**

1.  **Push to GitHub:** Upload `main.py`, `requirements.txt`, `Dockerfile`, and `.gitignore` to a new GitHub repository.
2.  **Create Space:** On Hugging Face, create a new Docker-based Space linked to your GitHub repo.
3.  **Set Secrets:** In the Space's settings, add your `NEON_DATABASE_URL`, `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, and `SHARED_SECRET`.
4.  **Deploy:** Hugging Face will automatically build and deploy your application, providing a permanent public URL.

**Final Integration**

1.  Go to your Exotel dashboard.
2.  Configure your phone number's call flow to connect to your new Hugging Face WebSocket URL (e.g., `wss://your-space-name.hf.space/listen`).
3.  In the connection settings, add a custom header `X-Auth-Token` with your `SHARED_SECRET` value for authentication.