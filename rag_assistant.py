"""
rag_assistant.py -- Part D: Policy Document Chunking, TF-IDF Retrieval, and Grounded QA

Tasks implemented:
  - Task 1: Policy document chunking (40 words, 10 overlap, 30 step) -> exactly 15 chunks.
  - Task 2: TF-IDF vector search and cosine similarity retrieval (retrieve(query, k=3)).
  - Task 3: Grounded Policy RAG Assistant (answer(question)).
  - Task 4: 14-question evaluation on policy_eval.csv and zero-overlap comparison.
"""

import glob
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm


POLICIES_DIR = os.path.join("data", "policies")
POLICY_EVAL_PATH = os.path.join("data", "raw", "policy_eval.csv")


# =============================================================================
# TASK 1: POLICY DOCUMENT CHUNKING
# =============================================================================
def chunk_policy_documents(overlap: int = 10, policies_dir: str = POLICIES_DIR) -> list[dict]:
    """
    Reads the five policy text files and splits each into chunks:
      - Chunk size: 40 words
      - Step size: 40 - overlap words (30 words for overlap=10)
      - Stops after the first chunk whose end reaches or passes the document length.

    Returns:
      List of dicts: [{'source': fn, 'category': cat, 'chunk_idx': i, 'text': '...'}]
    """
    assert os.path.exists(policies_dir), f"Missing policies directory: {policies_dir}"

    file_paths = sorted(glob.glob(os.path.join(policies_dir, "*.txt")))
    assert len(file_paths) == 5, f"Expected 5 policy files, found {len(file_paths)}"

    step = 40 - overlap
    chunks = []

    for path in file_paths:
        fn = os.path.basename(path)
        # Category is the prefix before '_policy.txt'
        category = fn.replace("_policy.txt", "")

        with open(path, "r", encoding="utf-8") as f:
            words = f.read().split()

        start = 0
        chunk_idx = 1
        while True:
            chunk_words = words[start : start + 40]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "source": fn,
                "category": category,
                "chunk_idx": chunk_idx,
                "text": chunk_text,
            })
            # Stop after the first chunk whose end reaches or passes document length
            if start + 40 >= len(words):
                break
            start += step
            chunk_idx += 1

    return chunks


# =============================================================================
# TASK 2: TF-IDF VECTOR RETRIEVER
# =============================================================================
class PolicyRetriever:
    """
    TF-IDF Vectorizer and Cosine Similarity retriever for policy chunks.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """
        Transforms query and returns top-k chunks ranked by cosine similarity descending.
        """
        if not query or not query.strip():
            # Handle empty query safely
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        # Top k indices sorted descending
        top_indices = sims.argsort()[::-1][:k]

        results = []
        for idx in top_indices:
            results.append({
                "source": self.chunks[idx]["source"],
                "category": self.chunks[idx]["category"],
                "score": round(float(sims[idx]), 4),
                "text": self.chunks[idx]["text"],
            })
        return results


# =============================================================================
# TASK 3: GROUNDED POLICY RAG ASSISTANT
# =============================================================================
def answer_policy_question(question: str, retriever: PolicyRetriever, k: int = 3) -> tuple[str, list[dict]]:
    """
    Answers a policy question using the POLICY ASSISTANT prompt trigger with mock_llm:
      1. Retrieves top-k chunks.
      2. Formats CONTEXT with source tags:
         CONTEXT:
         [<source file name>]
         <chunk text>
         QUESTION:
         <question>
      3. Sends messages to mock_llm.chat with system prompt containing 'POLICY ASSISTANT'.

    Returns:
      (model_answer, retrieved_chunks)
    """
    top_chunks = retriever.retrieve(question, k=k)

    context_lines = []
    for c in top_chunks:
        context_lines.append(f"[{c['source']}]\n{c['text']}")
    context_str = "\n".join(context_lines)

    user_msg = f"CONTEXT:\n{context_str}\nQUESTION:\n{question}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are an offline POLICY ASSISTANT for Instamart. "
                "Answer the customer question strictly using the provided policy context. "
                "If the context does not contain the answer, say you could not find the information."
            ),
        },
        {"role": "user", "content": user_msg},
    ]

    response = mock_llm.chat(messages)
    answer_text = response.choices[0].message.content or ""
    return answer_text, top_chunks


# =============================================================================
# TASK 4: EVALUATION & ZERO-OVERLAP COMPARISON
# =============================================================================
def evaluate_policy_rag(retriever: PolicyRetriever, eval_csv_path: str = POLICY_EVAL_PATH) -> tuple[pd.DataFrame, dict]:
    """
    Evaluates all 14 questions in policy_eval.csv:
      - 12 answerable questions: checks if expected_doc is in top 3 sources,
        and if expected_fact is a case-sensitive substring of the answer.
      - 2 unanswerable questions: checks if answer starts with 'I could not find'.
    """
    assert os.path.exists(eval_csv_path), f"Missing evaluation file: {eval_csv_path}"
    df_eval = pd.read_csv(eval_csv_path)

    records = []
    doc_matches = 0
    fact_matches = 0
    fallback_matches = 0

    for idx, row in df_eval.iterrows():
        q = row["question"]
        exp_doc = row["expected_doc"]
        exp_fact = row["expected_fact"] if pd.notna(row["expected_fact"]) else ""

        ans, chunks = answer_policy_question(q, retriever, k=3)
        retrieved_sources = [c["source"] for c in chunks]

        is_unanswerable = (exp_doc == "NONE")
        doc_pass = (exp_doc in retrieved_sources) if not is_unanswerable else True
        fact_pass = (exp_fact in ans) if not is_unanswerable else False
        fallback_pass = ans.startswith("I could not find") if is_unanswerable else False

        if not is_unanswerable:
            if doc_pass:
                doc_matches += 1
            if fact_pass:
                fact_matches += 1
        else:
            if fallback_pass:
                fallback_matches += 1

        records.append({
            "question": q,
            "expected_doc": exp_doc,
            "expected_fact": exp_fact,
            "retrieved_sources": ", ".join(retrieved_sources),
            "doc_match": doc_pass,
            "fact_match": fact_pass,
            "fallback_match": fallback_pass,
            "model_answer": ans,
        })

    df_results = pd.DataFrame(records)
    summary = {
        "doc_matches": doc_matches,
        "fact_matches": fact_matches,
        "fallback_matches": fallback_matches,
        "total_answerable": 12,
        "total_unanswerable": 2,
    }
    return df_results, summary


def run_chunking_comparison(eval_csv_path: str = POLICY_EVAL_PATH):
    """
    Compares RAG performance between:
      1. Default chunking (40 words, 10 overlap) -> 15 chunks
      2. Zero-overlap chunking (40 words, 0 overlap) -> 13 chunks
    Identifies specific questions that fail under zero-overlap chunking due to sentence splits.
    """
    print("\n" + "=" * 75)
    print("TASK 4: RAG EVALUATION & CHUNKING OVERLAP COMPARISON")
    print("=" * 75)

    # 1. Evaluate with Overlap = 10
    chunks_10 = chunk_policy_documents(overlap=10)
    retriever_10 = PolicyRetriever(chunks_10)
    df_res_10, summary_10 = evaluate_policy_rag(retriever_10, eval_csv_path)

    # 2. Evaluate with Overlap = 0
    chunks_0 = chunk_policy_documents(overlap=0)
    retriever_0 = PolicyRetriever(chunks_0)
    df_res_0, summary_0 = evaluate_policy_rag(retriever_0, eval_csv_path)

    doc_10 = f"{summary_10['doc_matches']}/12"
    doc_0 = f"{summary_0['doc_matches']}/12"
    fact_10 = f"{summary_10['fact_matches']}/12"
    fact_0 = f"{summary_0['fact_matches']}/12"
    fallback_10 = f"{summary_10['fallback_matches']}/2"
    fallback_0 = f"{summary_0['fallback_matches']}/2"

    print("\nRAG Evaluation Summary Table (14 Questions):")
    print(f"  {'Metric':<36}  {'Overlap = 10 (Default)':>22}  {'Overlap = 0 (No Overlap)':>25}")
    print(f"  {'-'*36}  {'-'*22}  {'-'*25}")
    print(f"  {'Total Chunks Generated':<36}  {len(chunks_10):>22d}  {len(chunks_0):>25d}")
    print(f"  {'Answerable Doc Match (Top 3)':<36}  {doc_10:>22}  {doc_0:>25}")
    print(f"  {'Answerable Fact Match':<36}  {fact_10:>22}  {fact_0:>25}")
    print(f"  {'Out-of-Domain Fallback Match':<36}  {fallback_10:>22}  {fallback_0:>25}")

    # Checkpoint validations for Overlap = 10
    assert summary_10["doc_matches"] == 12, f"Expected 12/12 doc matches, got {summary_10['doc_matches']}"
    assert summary_10["fact_matches"] == 12, f"Expected 12/12 fact matches, got {summary_10['fact_matches']}"
    assert summary_10["fallback_matches"] == 2, f"Expected 2/2 fallback matches, got {summary_10['fallback_matches']}"
    print("\n>> Checkpoint [Overlap=10 RAG Evaluation: 12/12 Doc, 12/12 Fact, 2/2 Fallback]: PASS")

    # Inspect failures under Overlap = 0
    failed_zero_overlap = df_res_0[
        (~df_res_0["fact_match"]) & (df_res_0["expected_doc"] != "NONE")
    ]
    print(f"\nZero-Overlap Degradation Analysis ({len(failed_zero_overlap)} question(s) failed):")
    for idx, r in failed_zero_overlap.iterrows():
        print(f"  - Failed Question: \"{r['question']}\"")
        print(f"    Expected Fact:   \"{r['expected_fact']}\"")
        print(f"    Model Answer:    \"{r['model_answer']}\"")
        print("    Failure Reason:  The contiguous sentence containing the fact was split across")
        print("                     the 40-word boundary, leaving the model with an incomplete clause.\n")

    assert len(failed_zero_overlap) >= 1, "Expected at least one failure under zero-overlap chunking"
    print(">> Checkpoint [Zero-Overlap Degradation Demonstrated]: PASS")

    # Save detailed evaluation to outputs/
    os.makedirs("outputs", exist_ok=True)
    df_res_10.to_csv("outputs/rag_evaluation_overlap10.csv", index=False)
    df_res_0.to_csv("outputs/rag_evaluation_overlap0.csv", index=False)
    print("Saved evaluation results to outputs/rag_evaluation_overlap10.csv and overlap0.csv")

    return df_res_10, df_res_0


# =============================================================================
# MAIN VALIDATION RUNNER
# =============================================================================
def main():
    print("=" * 75)
    print("PART D: POLICY DOCUMENT CHUNKING & RAG ASSISTANT")
    print("=" * 75)

    # Task 1: Document Chunking
    chunks = chunk_policy_documents(overlap=10)
    print(f"Total Policy Chunks (Overlap=10): {len(chunks)} (Expected: 15)")

    category_counts = {}
    for c in chunks:
        category_counts[c["category"]] = category_counts.get(c["category"], 0) + 1

    print("\nChunk Distribution by Policy Category:")
    for cat in ["refund", "delivery", "escalation", "payment", "privacy"]:
        count = category_counts.get(cat, 0)
        print(f"  - {cat:<12}: {count} chunks")

    assert len(chunks) == 15, f"Expected 15 total chunks, got {len(chunks)}"
    assert category_counts.get("refund") == 4, f"Expected 4 refund chunks, got {category_counts.get('refund')}"
    assert category_counts.get("delivery") == 3, f"Expected 3 delivery chunks, got {category_counts.get('delivery')}"
    assert category_counts.get("escalation") == 3, f"Expected 3 escalation chunks, got {category_counts.get('escalation')}"
    assert category_counts.get("payment") == 3, f"Expected 3 payment chunks, got {category_counts.get('payment')}"
    assert category_counts.get("privacy") == 2, f"Expected 2 privacy chunks, got {category_counts.get('privacy')}"
    print(">> Task 1 Checkpoint [15 Chunks: 4 refund, 3 delivery, 3 escalation, 3 payment, 2 privacy]: PASS")

    # Task 2: Vector Search & Benchmark Query
    retriever = PolicyRetriever(chunks)
    test_query = "refund policy late delivery"
    top3_res = retriever.retrieve(test_query, k=3)
    print(f"\nTask 2 Retrieval Benchmark Query: \"{test_query}\"")
    for i, res in enumerate(top3_res, 1):
        print(f"  {i}. Source: {res['source']:<20} Score: {res['score']:<8} Text preview: {res['text'][:60]}...")

    top3_sources = [r["source"] for r in top3_res]
    assert all(s == "refund_policy.txt" for s in top3_sources), (
        f"Expected all top 3 sources to be 'refund_policy.txt', got {top3_sources}"
    )
    print(">> Task 2 Checkpoint [Top 3 chunks for 'refund policy late delivery' are from refund_policy.txt]: PASS")

    # Tasks 3 & 4: RAG Assistant, 14-Question Evaluation, and Zero-Overlap Comparison
    run_chunking_comparison()


if __name__ == "__main__":
    main()
