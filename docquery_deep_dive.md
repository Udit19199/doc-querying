# docquery.py Deep Dive

## System Overview

This is a **document-based Question Answering system** built with Streamlit that compares two RAG (Retrieval-Augmented Generation) pipelines side-by-side (Set A vs Set B). It allows users to upload documents, configure different embedding models/chunking strategies/index types, ask questions, and evaluate responses using RAGAS metrics.

---

## Imports Breakdown

### UI & Web Framework
- **`streamlit as st`** - Web UI framework for building interactive data apps
- **`dotenv`** - Loads environment variables from `.env` file

### Document Processing
- **`PyPDF2.PdfReader`** - Parses PDF files, extracts text from pages
- **`DocxDocument`** - Reads Word documents (.docx)
- **`CharacterTextSplitter`** / **`RecursiveCharacterTextSplitter`** - Splits text into chunks for vectorization

### AI/ML Libraries
- **`numpy as np`** - Numerical computing, array operations for embeddings
- **`sentence_transformers`** - Provides the MiniLM model for local embeddings
- **`faiss`** - Facebook AI Similarity Search (vector database)
- **`sklearn.metrics.pairwise.cosine_similarity`** - Computes embedding similarity scores
- **`difflib`** / **`re`** - Fuzzy matching, text comparison

### Google Generative AI
- **`google.generativeai`** - Core Gemini API for embeddings and chat
- **`langchain_google_genai`** - LangChain integration for Google models
- **`ChatGoogleGenerativeAI`** - LLM wrapper for Gemini chat completion
- **`GoogleGenerativeAIEmbeddings`** - Creates embeddings using Gemini

### LangChain Framework
- **`HuggingFaceEmbeddings`** - Base class for HF embeddings
- **`OpenAIEmbeddings`** - OpenAI's text embeddings
- **`FAISS`** - Vector store wrapper (langchain_community)
- **`InMemoryDocstore`** - In-memory document storage
- **`Document`** - LangChain document object with `.page_content`
- **`load_qa_chain`** - Loads question-answering chain
- **`PromptTemplate`** - Templates for LLM prompts
- **`CharacterTextSplitter`** - Basic chunking

### Evaluation
- **`ragas.metrics`** - RAG evaluation metrics (answer_relevancy, faithfulness, context_precision, context_recall)
- **`ragas.evaluate`** - Runs RAGAS evaluation
- **`datasets.Dataset`** - HuggingFace dataset format for RAGAS

### PDF Report Generation
- **`reportlab`** - PDF generation library
- **`SimpleDocTemplate`** - PDF document builder
- **`Paragraph`** / **`Spacer`** / **`Table`** / **`TableStyle`** - PDF elements
- **`BytesIO`** - In-memory byte streams for PDF download

---

## Class: SafeHuggingFaceEmbeddings

**Lines 49-56** | Extends `HuggingFaceEmbeddings`

**Purpose**: Wrapper that forces CPU execution (avoids GPU issues on some systems)

**How it works**:
- Overrides `__init__` to set `model_kwargs["device"] = "cpu"`
- Creates a `SentenceTransformer` directly (the underlying HF model)
- Passes it to parent class via `self.client = model`

**Why needed**: Some environments lack GPU support, and the default HF embeddings may auto-detect and fail

---

## Global Setup

**Lines 58-66**

```python
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
EVAL_BUFFER = []
```

**Purpose**: Initializes Google API credentials and creates a global buffer for evaluation logging

**`EVAL_BUFFER`**: List that accumulates (question, contexts, answer) tuples for batch evaluation

---

## Function: log_for_eval()

**Lines 67-73**

**Purpose**: Appends Q/A/context to evaluation buffer for later RAGAS analysis

**Params**:
- `question` (str)
- `contexts` (list): Document objects
- `answer` (str)

**Logic**: Converts any LangChain Documents to their `.page_content`, stores in global `EVAL_BUFFER`

---

## Function: run_eval()

**Lines 76-78**

**Purpose**: Placeholder/entry point for running RAGAS evaluation on the accumulated buffer

**Calls**: `build_eval_dataset()` → creates dataset from `EVAL_BUFFER` → `evaluate()` with RAGAS metrics

---

## Function: get_file_text()

**Lines 82-99**

**Purpose**: Extracts raw text from uploaded files (PDF/DOCX/TXT)

**Params**:
- `uploaded_files` (list): Streamlit file upload objects

**Logic per file type**:
- **PDF**: Uses `PdfReader`, extracts text from each page
- **DOCX**: Uses `python-docx`, reads all paragraphs
- **TXT**: Reads bytes, decodes as UTF-8

**Returns**: Concatenated text string from all files

---

## Function: get_text_chunks()

**Lines 101-119**

**Purpose**: Splits raw text into smaller chunks for vectorization

**Params**:
- `text` (str): Raw text
- `chunk_strategy` (str): One of "Recursive", "Fixed-Size", "Paragraph"

**Chunking strategies**:
1. **Recursive** (default): Splits by "\n\n", "\n", " ", "" hierarchically (500 chars, 100 overlap)
2. **Fixed-Size**: Splits by newlines first, then by 400 chars with 100 overlap
3. **Paragraph**: Splits on double newlines (`\n\n`), returns whole paragraphs

**Why chunking matters**: LLMs have context limits; documents must be split into retrievable pieces

**Returns**: List of text chunks (strings)

---

## Function: build_and_save_faiss()

**Lines 121-186**

**Purpose**: Creates a FAISS vector index from text chunks and saves it to disk

**Params**:
- `text_chunks` (list): List of text strings
- `embedding_model` (str): Which embedding model to use
- `index_type` (str): "Flat", "IVF", or "HNSW"
- `folder_name` (str): Directory to save index

**Steps**:

1. **Get embeddings** (line 122): Calls `get_embeddings()` to get the embedding function
2. **Vectorize chunks** (line 123): `embeddings.embed_documents(text_chunks)` → numpy array
3. **Create FAISS index** based on type:
   - **Flat** (line 129): Exact nearest neighbor search, slow for large datasets
   - **IVF** (lines 130-135): Approximate search with inverted file index, faster
   - **HNSW** (lines 136-138): Hierarchical Navigable Small World graph, best speed/accuracy tradeoff
4. **Build docstore** (lines 144-147): Maps UUIDs to LangChain Documents
5. **Create FAISS store** (lines 149-181): Handles different LangChain versions (embedding vs embedding_function parameter)
6. **Save to disk** (lines 184-186): Overwrites existing folder, saves index + docstore

**FAISS Index Types Explained**:
- **Flat**: Brute-force search through all vectors (O(n)), perfect accuracy
- **IVF**: Clusters vectors, searches only relevant clusters (faster, slight accuracy loss)
- **HNSW**: Graph-based navigation, excellent speed/accuracy balance

---

## Function: _unwrap_index()

**Lines 189-215**

**Purpose**: Defensive utility to extract raw FAISS index from wrapper objects

**Problem**: FAISS indices can be wrapped (e.g., `IndexIDMap`, `IndexShards`) which hide the actual search index

**Logic**: Recursively checks for:
- `.index` attribute (IndexIDMap, IndexIDMap2)
- `.shards`, `.indexes`, `.sub_indexes`, `.components` attributes

**Returns**: The innermost raw `faiss.Index` object

**Why needed**: For configuration (like setting `nprobe` on IVF indexes), you need the actual index

---

## Function: get_conversational_chain()

**Lines 218-282**

**Purpose**: Creates a LangChain QA chain with the LLM (Gemini)

**Params**: None (reads from `st.session_state`)

**Session state used**:
- `prompt_style`: "Detailed Answer", "Concise Summary", "Bullet Points", "Explain Like I'm 5"
- `temperature`: LLM creativity (0.0-1.0)
- `generating_model`: e.g., "models/gemini-2.5-flash"

**Prompt templates** (lines 221-267):
- **Detailed Answer**: "Prioritize clarity, structure, reasoning"
- **Concise Summary**: Brief, clear summary
- **Bullet Points**: Structured bullet points
- **ELI5**: Simple explanation

**Chain creation** (line 281):
```python
chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
```

**"stuff" chain type**: Stuffs all retrieved documents into the prompt context (simple, works for small contexts)

**Returns**: LangChain QA chain (callable)

---

## Function: get_similarity_score()

**Lines 284-366**

**Purpose**: Computes 0-1 similarity between answer and retrieved contexts

**Multi-strategy approach** (robust fallback chain):

**Strategy 1: Exact substring** (lines 311-312)
- If answer is substring of context → score = 1.0

**Strategy 2: Exact sentence match** (lines 315-320)
- Splits contexts into sentences
- If answer matches any sentence → score = 1.0

**Strategy 3: Embedding similarity** (lines 322-351)
- Tries HuggingFace MiniLM first (local, fast)
- Falls back to Google embeddings if HF fails or explicitly requested
- Computes cosine similarity between context embedding and answer embedding
- Normalizes negative scores: `(score + 1) / 2`

**Strategy 4: Fuzzy matching** (lines 353-362)
- Uses `difflib.SequenceMatcher` for character-level similarity
- Best ratio across all contexts → final score

**Returns**: Float 0.0-1.0

**Used by**: RAGAS fallback when ground truth isn't available

---

## Function: extract_reference_from_contexts()

**Lines 368-391**

**Purpose**: Picks the "best" sentence from contexts as a pseudo-reference for RAGAS

**Logic**:
1. Splits all contexts into sentences (by `.!?`)
2. Scores each sentence against answer using fuzzy matching
3. Returns the highest-scoring sentence

**Why needed**: RAGAS metrics need a reference answer; when none exists, this provides a cheap alternative

**Fallback**: If no sentences found, returns first 400 chars of joined contexts

---

## Function: save_report_as_pdf()

**Lines 395-438**

**Purpose**: Exports single query evaluation as PDF

**Params**:
- `filename` (str): Output filename
- `eval_dict` (dict): Contains question, answer, contexts
- `metrics` (dict): RAGAS metric scores

**PDF elements**:
- Title: "Query Evaluation Report"
- Question, Answer, Retrieved Contexts
- Metrics table (styled with grey header, grid lines)

**Uses ReportLab**: Python PDF generation library

---

## Function: compare_across_models()

**Lines 441-498**

**Purpose**: Compares response quality across different embedding models

**Models tested**:
- Google (Gemini) embeddings
- MiniLM (HuggingFace) local embeddings
- OpenAI embeddings (if API key available)

**For each model**:
1. Embeds text chunks
2. Creates FAISS index (Flat/IVF/HNSW)
3. Creates LangChain FAISS vectorstore
4. Similarity search for user question
5. Runs QA chain to get answer
6. Displays in column

**Note**: This function is defined but seems less used than `compare_between_sets()`

---

## Function: rerank_chunks_llm()

**Lines 501-536**

**Purpose**: Re-ranks retrieved chunks using LLM scoring (not embeddings)

**Params**:
- `question` (str)
- `chunks` (list): Retrieved Document objects
- `top_k` (int): How many chunks to return

**LLM Scoring Prompt** (lines 512-523):
```
Rate relevance 0-10:
Chunk: "{chunk.page_content}"
Question: {question}
Respond with single number.
```

**Process**:
1. For each chunk, invoke LLM with scoring prompt
2. Extract numeric score from response
3. Sort chunks by score descending
4. Return top_k chunks

**Why**: Sometimes semantic similarity isn't enough; LLM can better assess relevance

---

## Function: compare_between_sets()

**Lines 539-1023**

**Purpose**: **Core function** - Runs comparison between Set A and Set B

**High-level flow**:

```
For each set (A, B):
  1. Load FAISS index from disk
  2. Unwrap index (configure IVF/HNSW parameters)
  3. Similarity search with question
  4. Optional: LLM re-ranking
  5. Run QA chain → get answer
  6. Calculate similarity score
  7. Run RAGAS evaluation
  8. Display results in column
  
Export: Generate PDF report with both sets
```

**Key subsections**:

**Index Loading** (lines 548-576):
- Loads from `faiss_setA` or `faiss_setB` folder
- Unwraps index to configure search parameters
- Sets `nprobe` for IVF, `efSearch` for HNSW

**RAGAS Evaluation** (lines 637-734):
- Creates HuggingFace Dataset from eval data
- Wraps LangChain LLM/embeddings for RAGAS
- Handles multiple RAGAS API signatures (varies by version)
- Falls back gracefully if quota exceeded

**Metrics displayed**:
- **Answer Relevancy**: Does answer address the question?
- **Faithfulness**: Does answer stick to retrieved context?
- **Context Precision**: Was retrieved context relevant?
- **Context Recall**: Was relevant context retrieved?

**Export** (lines 1015-1022):
- Generates PDF with both sets' questions, answers, contexts, metrics
- Download button for PDF report

---

## Function: get_embeddings()

**Lines 1025-1034**

**Purpose**: Factory function to get embedding model by name

**Supported models**:
- `"Google (Gemini)"` → `GoogleGenerativeAIEmbeddings(model="models/embedding-001")`
- `"MiniLM (HuggingFace)"` → `SafeHuggingFaceEmbeddings(...)`
- `"OpenAI"` → `OpenAIEmbeddings()`

**Returns**: Embedding function (callable with `.embed_documents()` and `.embed_query()`)

---

## Function: parameter_topbar()

**Lines 1037-1142**

**Purpose**: Creates the main UI configuration panel

**Sets default session state** (lines 1039-1057):
- Set A: Google embeddings, Flat index, Recursive chunking
- Set B: MiniLM embeddings, Flat index, Recursive chunking
- Common: top_k=5, temperature=0.5, prompt_style="Detailed Answer"

**UI Layout**:
1. **Set A column**: Index type, Chunking strategy, Embedding model
2. **Set B column**: Same 3 parameters
3. **Common parameters row**: Top-K slider, Temperature slider, Prompt style dropdown, Re-ranking checkbox
4. **Model info**: Generating model, Vector DB (display-only)

**Button**: "Confirm Parameter Sets" → sets `compare_ready = True`

---

## Function: sidebar_upload_section()

**Lines 1147-1191**

**Purpose**: Sidebar for file upload and document processing

**UI elements**:
- File uploader (PDF/DOCX/TXT, multiple files)
- "Submit & Process" button

**On submit**:
1. Validates parameters are confirmed and files uploaded
2. Extracts text from files
3. **Process Set A**: Chunks with Set A strategy → `build_and_save_faiss()` → `faiss_setA`
4. **Process Set B**: Chunks with Set B strategy → `build_and_save_faiss()` → `faiss_setB`
5. Saves raw text to session state

---

## Function: main()

**Lines 1193-1214**

**Purpose**: App entry point, orchestrates all components

**Flow**:
1. Set page config (wide layout)
2. Display title
3. Render parameter topbar
4. Render sidebar upload section
5. Wait for user question input
6. When question submitted + documents loaded → call `compare_between_sets()`

---

## Complete Data Flow

```
User uploads files
        ↓
parameter_topbar() → Set A and Set B configurations
        ↓
sidebar_upload_section()
    ├── get_file_text() → extract raw text
    ├── get_text_chunks() → split into chunks (Set A strategy)
    ├── build_and_save_faiss() → vectorize + FAISS index → faiss_setA
    ├── get_text_chunks() → split into chunks (Set B strategy)
    └── build_and_save_faiss() → vectorize + FAISS index → faiss_setB
        ↓
User enters question
        ↓
compare_between_sets(question)
    ├── Load faiss_setA → similarity_search() → docs_A
    ├── Load faiss_setB → similarity_search() → docs_B
    ├── Optional: rerank_chunks_llm() for each set
    ├── get_conversational_chain() → LLM chain
    ├── For each set:
    │   ├── chain({"input_documents": docs, "question}) → answer
    │   ├── get_similarity_score() → relevance score
    │   ├── run_ragas_on() → evaluation metrics
    │   └── extract_reference_from_contexts() → pseudo-reference
    └── Display side-by-side comparison
        ↓
User clicks "Download Report"
        ↓
generate_pdf_bytes_from_payload() → PDF download
```

---

## Key Technologies Explained

### RAG (Retrieval-Augmented Generation)
- Retrieve relevant document chunks based on question
- Augment prompt with retrieved context
- Generate answer using LLM

### Embeddings
- Convert text to dense vectors (300+ dimensions)
- Similar texts have similar vectors (high cosine similarity)
- Enable semantic search instead of keyword matching

### FAISS
- Efficiently searches millions of embedding vectors
- Supports exact (Flat) and approximate (IVF, HNSW) search
- Index maps vectors → document chunks

### RAGAS
- Evaluates RAG systems without human labels
- Uses LLM-as-judge for automatic metric calculation
- Metrics: Answer Relevancy, Faithfulness, Context Precision/Recall
