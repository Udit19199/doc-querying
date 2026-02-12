# Deep Dive: `compare_between_sets` Function

The `compare_between_sets` function (lines 626-1182) is the core comparison engine for your RAG system. It compares two RAG configurations side-by-side to help you determine which embedding/index/chunking combination works best for your documents.

---

## Section 1: Setup & Initialization (Lines 626-645)

```python
def compare_between_sets(user_question=None):
    cols = st.columns(2)
    eval_data = []
    chain = get_conversational_chain()
    for i, (label, set_key, faiss_folder) in enumerate(
        [
            ("Set A", "setA", "faiss_setA"),
            ("Set B", "setB", "faiss_setB"),
        ]
    ):
        try:
            if set_key not in st.session_state:
                raise ValueError(f"{set_key} not configured in session_state")
            emb_model_name = st.session_state[set_key]["embedding_model"]
            emb_model = get_embeddings(emb_model_name)
            db = FAISS.load_local(
                faiss_folder, emb_model, allow_dangerous_deserialization=True
            )
            top_k = st.session_state.get("top_k", 5)
```

**What happens here:**

- `cols = st.columns(2)` creates two side-by-side Streamlit columns for visual comparison
- `eval_data = []` initializes an empty list to collect evaluation metrics for later display
- `chain = get_conversational_chain()` loads the QA chain (LLM + prompt template) once, reused for both sets
- The loop iterates over two configurations: Set A and Set B
  - `label`: Display name ("Set A" / "Set B")
  - `set_key`: Session state key ("setA" / "setB") containing config
  - `faiss_folder`: Folder name where FAISS index is saved
- Retrieves the embedding model name from session state and converts it to an actual embedding object via `get_embeddings()`
- Loads the FAISS index from disk with `allow_dangerous_deserialization=True` (required for FAISS)
- `top_k` defaults to 5 if not set in session state

---

## Section 2: Index Debug Info & Unwrapping (Lines 645-656)

```python
            # ---------- DEBUG: show index size and top_k requested ----------
            try:
                index_size = getattr(db.index, "ntotal", None)
            except Exception:
                index_size = None
            # fallback to docstore size
            try:
                docstore_size = len(getattr(db, "docstore")._dict)
            except Exception:
                docstore_size = None

            base_index = _unwrap_index(db.index)
```

**What happens here:**

- Attempts to get `ntotal` from the FAISS index (number of vectors indexed)
- Falls back to counting docstore entries if `ntotal` unavailable
- `_unwrap_index()` (defined at lines 242-279) recursively unwraps FAISS wrapper objects like `IndexIDMap`, `IndexIVFFlat`, `IndexHNSWFlat` to get the underlying raw index
- This is defensive code: FAISS often wraps indexes for functionality (e.g., ID mapping), and you need the base index for operations like setting `nprobe`

---

## Section 3: Index-Specific Tuning (Lines 658-664)

```python
            if isinstance(base_index, faiss.IndexIVFFlat):
                nlist = getattr(base_index, "nlist", None) or 100
                probe = max(2, min(nlist, top_k * 2))
                base_index.nprobe = probe
            elif isinstance(base_index, faiss.IndexHNSWFlat):
                base_index.hnsw.efSearch = max(50, top_k * 4)
```

**What happens here:**

- **For IVF indexes**: Sets `nprobe` (number of Voronoi cells to search)
  - Formula: `probe = max(2, min(nlist, top_k * 2))`
  - Ensures at least 2 cells, at most all cells (`nlist`), scaled by `top_k`
  - More probes = more accurate but slower
- **For HNSW indexes**: Sets `efSearch` (search depth)
  - Formula: `efSearch = max(50, top_k * 4)`
  - Minimum 50, scales with `top_k`
  - Higher values = better recall, more computation

---

## Section 4: Similarity Search (Lines 666-674)

```python
            query = user_question or ""
            docs = db.similarity_search(query, k=top_k, fetch_k=max(20, top_k * 5))

            if st.session_state.get("enable_rerank"):
                with cols[i]:
                    st.info(f"Re-ranking top {len(docs)} chunks using Ollama...")
                docs = rerank_chunks_llm(
                    user_question, docs, top_k=st.session_state.get("top_k", 5)
                )
```

**What happens here:**

- `query = user_question or ""` handles None gracefully
- `db.similarity_search(query, k=top_k, fetch_k=max(20, top_k * 5))`
  - `k=top_k`: Return exactly `top_k` results to the LLM
  - `fetch_k=max(20, top_k * 5)`: Fetch more candidates from vector DB before final selection
  - This helps when reranking is enabled (more candidates to choose from)
- If `enable_rerank` is True:
  - Shows info message in the current column
  - Calls `rerank_chunks_llm()` which uses the LLM to score each chunk's relevance (0-10) and returns top-k

---

## Section 5: LLM Answer Generation (Lines 675-681)

```python
            start_time = time.time()
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True,
            )
            end_time = time.time()
            latency = round(end_time - start_time, 2)

            answer_text = response["output_text"]
```

**What happens here:**

- `start_time` captures when retrieval ends and generation begins
- `chain.invoke()` passes the retrieved documents and question to the LLM
  - `return_only_outputs=True` ensures we get only the generated text, not intermediate outputs
- `end_time` captures when generation finishes
- `latency = end_time - start_time` measures generation time in seconds
- `answer_text = response["output_text"]` extracts the LLM's answer

---

## Section 6: Session State Storage (Lines 683-696)

```python
            if label == "Set A":
                st.session_state.eval_A = {
                    "question": user_question,
                    "contexts": [d.page_content for d in docs],  # strings for ragas
                    "answer": answer_text,
                }
            elif label == "Set B":
                st.session_state.eval_B = {
                    "question": user_question,
                    "contexts": [d.page_content for d in docs],
                    "answer": answer_text,
                }
```

**What happens here:**

- Stores evaluation data in `st.session_state` for persistence across reruns
- `eval_A` and `eval_B` dictionaries contain:
  - `question`: The user's query
  - `contexts`: List of document page contents (strings, not Document objects) for RAGAS
  - `answer`: Generated response

---

## Section 7: Logging & Similarity Scoring (Lines 698-703)

```python
            # Log for evaluation
            log_for_eval(user_question, docs, answer_text)
            similarity = get_similarity_score(answer_text, docs, emb_model_name)
            cols[i].markdown(f"**{label}**")
            cols[i].write(answer_text)
```

**What happens here:**

- `log_for_eval()` appends to `EVAL_BUFFER` for batch evaluation tracking
- `get_similarity_score()` computes relevance between answer and retrieved contexts
  - Returns 0-1 float (see function at lines 351-439 for full logic)
- Displays the set label and generated answer in the respective column

---

## Section 8: Evaluation Data Collection (Lines 704-714)

```python
            eval_data.append(
                {
                    "Set": label,
                    "Embedding": emb_model_name,
                    "Index": st.session_state[set_key]["index_type"],
                    "Chunking": st.session_state[set_key]["chunk_strategy"],
                    "Answer Length": len(answer_text),
                    "Latency (s)": latency,
                    "Relevance Score": similarity,
                }
            )
```

**What happens here:**

- Collects metadata about this run into `eval_data`
- This creates a row that could be displayed as a table showing:
  - Which embedding model was used
  - Index type (Flat/IVF/HNSW)
  - Chunking strategy
  - Answer length (character count)
  - Generation latency
  - Embedding-based similarity score

---

## Section 9: Error Handling for Each Set (Lines 715-716)

```python
        except Exception as e:
            cols[i].error(f"{label} failed: {str(e)}")
```

**What happens here:**

- If ANY step in the try block fails for a set, displays an error in that column
- Continues to process the other set even if one fails

---

## Section 10: RAGAS Header (Lines 718-719)

```python
    # === RAGAS evaluation (unchanged logic, but we store results into session_state so we can export them) ===
    st.markdown("RAGAS Evaluation")
```

**What happens here:**

- Section divider for RAGAS-based evaluation
- RAGAS is a framework for evaluating RAG systems with LLM-based metrics

---

## Section 11: Column Setup & Helper Functions (Lines 721-731)

```python
    colA, colB = st.columns(2)

    # Helper to format metric
    def fmt_metric(val):
        return "—" if pd.isna(val) else f"{float(val):.2f}"

    def my_llm_factory():
        return ChatOllama(model="llama3.2", temperature=0.0)

    def my_embedding_factory():
        return OllamaEmbeddings(model="nomic-embed-text")
```

**What happens here:**

- Creates two columns for RAGAS results (Set A left, Set B right)
- `fmt_metric()` formats values: shows "—" for NaN, otherwise 2 decimal places
- `my_llm_factory()` creates an LLM for RAGAS evaluation (llama3.2, temp=0 for determinism)
- `my_embedding_factory()` creates embeddings for RAGAS (nomic-embed-text)

---

## Section 12: `run_ragas_on` Function Definition (Lines 733-846)

This is a nested helper function that runs RAGAS evaluation for a single set.

### 12a: Reference Extraction (Lines 733-738)

```python
    def run_ragas_on(eval_dict, label, container):
        ref_text = eval_dict.get("reference")
        if not ref_text:
            ref_text = extract_reference_from_contexts(
                eval_dict.get("answer", ""), eval_dict.get("contexts", [])
            )
```

- If no ground-truth reference exists, extracts the best-matching sentence from contexts as a "reference"
- `extract_reference_from_contexts()` (lines 442-467) uses fuzzy matching to find the most relevant sentence

### 12b: Dataset Creation (Lines 740-747)

```python
        ds = Dataset.from_dict(
            {
                "question": [eval_dict.get("question", "")],
                "contexts": [eval_dict.get("contexts", [])],
                "answer": [eval_dict.get("answer", "")],
                "reference": [ref_text],
            }
        )
```

- Creates a HuggingFace `Dataset` object with one row for RAGAS evaluation
- RAGAS expects these columns: question, contexts, answer, reference

### 12c: LLM/Embedding Initialization (Lines 749-763)

```python
        try:
            eval_llm = ChatOllama(
                model=st.session_state.get("generating_model", "llama3.2"),
                temperature=0.0,
            )
        except Exception as e:
            container.error(f"Could not instantiate evaluation LLM: {e}")
            return None

        try:
            eval_emb = OllamaEmbeddings(model="nomic-embed-text")
        except Exception as e:
            container.error(f"Could not instantiate evaluation embeddings: {e}")
            return None
```

- Creates evaluation LLM and embeddings for RAGAS metrics
- Returns `None` if either fails, with error shown in the container

### 12d: Wrapper Compatibility (Lines 765-783)

```python
        llm_for_ragas = None
        emb_for_ragas = None
        try:
            from ragas.llms import LangchainLLMWrapper

            llm_for_ragas = LangchainLLMWrapper(eval_llm)
        except Exception:
            try:
                from ragas.llms import LangchainLLM

                llm_for_ragas = LangchainLLM(eval_llm)
            except Exception:
                llm_for_ragas = eval_llm

        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper

            emb_for_ragas = LangchainEmbeddingsWrapper(eval_emb)
        except Exception:
            emb_for_ragas = eval_emb
```

- Handles different RAGAS versions with different wrapper APIs
- Tries newer `LangchainLLMWrapper` first, falls back to `LangchainLLM`, then raw object
- Same pattern for embeddings

### 12e: Signature Detection (Lines 785-798)

```python
        sig = None
        try:
            sig = inspect.signature(evaluate)
            params = set(sig.parameters.keys())
        except Exception:
            params = set()

        metrics_list = [
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        ]
```

- Uses `inspect.signature()` to detect which parameters the current RAGAS `evaluate()` function accepts
- Different RAGAS versions have different APIs (some use `llm`, some use `llm_factory`)

### 12f: Multi-Signature Evaluation Call (Lines 800-846)

```python
        try:
            if "llm" in params or "embeddings" in params:
                results = evaluate(
                    ds,
                    metrics=metrics_list,
                    llm=llm_for_ragas,
                    embeddings=emb_for_ragas,
                )
                return results

            if "llm_or_chain_factory" in params:
                results = evaluate(
                    ds, llm_or_chain_factory=llm_for_ragas, metrics=metrics_list
                )
                return results

            if "llm_factory" in params or "embedding_factory" in params:
                results = evaluate(
                    ds,
                    metrics=metrics_list,
                    llm_factory=(lambda: llm_for_ragas),
                    embedding_factory=(lambda: emb_for_ragas),
                )
                return results

            try:
                results = evaluate(ds, metrics=metrics_list, llm=llm_for_ragas)
                return results
            except Exception:
                results = evaluate(ds, metrics=metrics_list)
                return results

        except ResourceExhausted as e:
            container.warning(
                f"Evaluation LLM quota issue for {label}: {str(e)}. Falling back to evaluate() without custom models."
            )
            try:
                results = evaluate(ds, metrics=metrics_list)
                return results
            except Exception as e2:
                container.error(f"Evaluation completely failed for {label}: {e2}")
                return None
        except TypeError as e:
            container.error(f"Evaluation failed due to unexpected signature: {e}")
            return None
        except Exception as e:
            container.error(f"Evaluation failed for {label}: {e}")
            return None
```

- Tries multiple API signatures in order of specificity
- Handles `ResourceExhausted` (API quota issues) gracefully with fallback
- Returns `None` on complete failure, with error message displayed

---

## Section 13: Set A Evaluation Display (Lines 848-935)

```python
    with colA:
        if "eval_A" in st.session_state:
            container = st.container()
            container.subheader("Set A")
            container.markdown(f"**Question:** {st.session_state.eval_A['question']}")
            with container.expander("Retrieved Chunks (A)", expanded=False):
                # ... displays chunks with preview and "show all" options ...
            container.markdown("**Answer:**")
            container.write(st.session_state.eval_A["answer"])

            res_a = run_ragas_on(st.session_state.eval_A, "Set A", container)
            st.session_state["last_ragas_A"] = res_a

            if res_a is not None:
                df_a = res_a.to_pandas()
                # ... extracts metrics and displays them ...
```

**What happens here:**

- Checks if `eval_A` exists in session state
- Displays:
  - The question asked
  - Retrieved chunks in an expander (with single-chunk preview + "show all" option)
  - Generated answer
  - RAGAS metrics (Answer Relevancy, Faithfulness, Context Precision, Context Recall)

---

## Section 14: Set B Evaluation Display (Lines 937-1025)

```python
    with colB:
        if "eval_B" in st.session_state:
            container = st.container()
            container.subheader("Set B")
            # ... same structure as Set A ...
```

**What happens here:**

- Identical structure to Set A, but for Set B
- Uses `eval_B` from session state
- Runs `run_ragas_on(st.session_state.eval_B, "Set B", container)`

---

## Section 15: Export Section (Lines 1027-1181)

### 15a: Export Selection (Lines 1027-1043)

```python
    st.markdown("---")
    st.subheader("Export / Download Report")

    if "export_choice" not in st.session_state:
        st.session_state["export_choice"] = "Both sets"

    export_choice = st.selectbox(
        "Which set do you want to export?",
        ["Set A", "Set B", "Both sets"],
        index=["Set A", "Set B", "Both sets"].index(st.session_state["export_choice"]),
        key="export_dropdown",
    )
    st.session_state["export_choice"] = export_choice
```

- Dropdown to select which report(s) to export
- Persists selection in session state

### 15b: `build_metrics_from_ragas` Helper (Lines 1045-1077)

```python
    def build_metrics_from_ragas(ragas_result, eval_dict, embedding_model_name):
        fallback = get_similarity_score(
            eval_dict.get("answer", ""),
            eval_dict.get("contexts", []),
            embedding_model_name,
        )
        default_metrics = {
            "Faithfulness": fallback,
            "Answer Relevancy": fallback,
            "Context Precision": fallback,
            "Context Recall": fallback,
        }

        if ragas_result is None:
            return default_metrics
        try:
            df = ragas_result.to_pandas()

            def safe_val(col):
                if col in df.columns and not pd.isna(df.at[0, col]):
                    return float(df.at[0, col])
                else:
                    return fallback

            return {
                "Faithfulness": safe_val("faithfulness"),
                "Answer Relevancy": safe_val("answer_relevancy"),
                "Context Precision": safe_val("context_precision"),
                "Context Recall": safe_val("context_recall"),
            }
        except Exception:
            return default_metrics
```

- Creates a fallback score using embedding similarity
- If RAGAS result exists, extracts actual metric values; otherwise uses fallback

### 15c: `generate_pdf_bytes_from_payload` Helper (Lines 1079-1130)

```python
    def generate_pdf_bytes_from_payload(payload_dict):
        buf = BytesIO()
        styles = getSampleStyleSheet()
        story = []
        for label, (eval_d, metrics) in payload_dict.items():
            story.append(
                Paragraph(f"<b>{label} — Query Evaluation Report</b>", styles["Title"])
            )
            story.append(Spacer(1, 8))
            story.append(
                Paragraph(
                    f"<b>Question:</b> {eval_d.get('question', '')}", styles["Normal"]
                )
            )
            story.append(Spacer(1, 6))
            story.append(Paragraph("<b>Answer:</b>", styles["Heading3"]))
            story.append(Paragraph(eval_d.get("answer", ""), styles["Normal"]))
            story.append(Spacer(1, 8))
            story.append(Paragraph("<b>Retrieved Contexts:</b>", styles["Heading3"]))
            for i, ctx in enumerate(eval_d.get("contexts", [])):
                story.append(Paragraph(f"Context {i + 1}:", styles["Normal"]))
                story.append(Paragraph(ctx, styles["Normal"]))
                story.append(Spacer(1, 6))
            story.append(Spacer(1, 8))
            story.append(Paragraph("<b>Evaluation Metrics:</b>", styles["Heading3"]))
            table_data = [["Metric", "Score"]]
            for k, v in metrics.items():
                table_data.append(
                    [k, f"{v:.2f}" if isinstance(v, (int, float)) else str(v)]
                )
            table = Table(table_data, colWidths=[200, 200])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 12))
            story.append(PageBreak())

        doc = SimpleDocTemplate(buf, pagesize=(595.27, 841.89))
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
```

- Uses ReportLab to generate PDFs programmatically
- Iterates over selected sets, adding each to the PDF
- Returns PDF as bytes (not file)

### 15d: Payload Assembly & Download (Lines 1132-1181)

```python
    payload = {}

    if export_choice == "Set A":
        eval_A = st.session_state.get("eval_A")
        ragas_A = st.session_state.get("last_ragas_A")
        if eval_A:
            metrics_A = build_metrics_from_ragas(
                ragas_A, eval_A, st.session_state.setA["embedding_model"]
            )
            payload = {"Set A": (eval_A, metrics_A)}
        else:
            st.warning("No evaluation data available for Set A.")

    elif export_choice == "Set B":
        eval_B = st.session_state.get("eval_B")
        ragas_B = st.session_state.get("last_ragas_B")
        if eval_B:
            metrics_B = build_metrics_from_ragas(
                ragas_B, eval_B, st.session_state.setB["embedding_model"]
            )
            payload = {"Set B": (eval_B, metrics_B)}
        else:
            st.warning("No evaluation data available for Set B.")

    else:  # Both sets
        eval_A = st.session_state.get("eval_A")
        eval_B = st.session_state.get("eval_B")
        ragas_A = st.session_state.get("last_ragas_A")
        ragas_B = st.session_state.get("last_ragas_B")
        if eval_A and eval_B:
            metrics_A = build_metrics_from_ragas(
                ragas_A, eval_A, st.session_state.setA["embedding_model"]
            )
            metrics_B = build_metrics_from_ragas(
                ragas_B, eval_B, st.session_state.setB["embedding_model"]
            )
            payload = {"Set A": (eval_A, metrics_A), "Set B": (eval_B, metrics_B)}
        else:
            st.warning("Both Set A and Set B must be evaluated first before exporting.")

    if payload:
        pdf_bytes = generate_pdf_bytes_from_payload(payload)
        st.download_button(
            f"Download {export_choice} Report",
            data=pdf_bytes,
            file_name=f"{export_choice.replace(' ', '_')}_report.pdf",
            mime="application/pdf",
        )
```

- Assembles the payload based on export choice
- Creates PDF bytes and provides download button

---

## Summary

The `compare_between_sets` function is a comprehensive RAG evaluation tool that:

1. **Retrieves** documents using two different configurations
2. **Generates** answers from both configurations
3. **Scores** answers using both embedding similarity and RAGAS metrics
4. **Displays** results side-by-side for comparison
5. **Exports** detailed PDF reports

### Key Design Patterns

1. **Defensive Coding**: The function uses extensive try/except blocks to handle failures gracefully
2. **API Compatibility**: Multiple RAGAS API signatures are tried to support different versions
3. **Session State Persistence**: Results are stored in `st.session_state` to survive reruns
4. **Modular Helper Functions**: Nested functions like `run_ragas_on()` and `build_metrics_from_ragas()` improve code organization
5. **Visual Organization**: Streamlit columns, expanders, and containers organize content clearly

### Bug Note

There is a bug at lines 997-1000 in the Set B evaluation section where `df_a` is used instead of `df_b`:

```python
answer_rel = safe_get(df_a, "answer_relevancy", fallback_score)  # Should be df_b
faithful = safe_get(df_a, "faithfulness", fallback_score)        # Should be df_b
ctx_prec = safe_get(df_a, "context_precision", fallback_score)  # Should be df_b
ctx_rec = safe_get(df_a, "context_recall", fallback_score)       # Should be df_b
```

This causes Set B to display Set A's RAGAS metrics instead of its own.
