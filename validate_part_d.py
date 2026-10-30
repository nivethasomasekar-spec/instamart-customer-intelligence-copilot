"""
validate_part_d.py -- Part D Comprehensive Validation Suite

Independently validates the deliverables, data files, and execution mechanics
of Part D: Policy RAG Assistant and Autonomous Refund Agent.

Asserts 10 rigorous checkpoints against actual files and live execution:
  1. Policy Document Chunking: exactly 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2)
  2. Overlap-10 RAG Evaluation: 12/12 doc matches, 12/12 fact matches, 2/2 out-of-domain fallbacks
  3. Overlap-0 Degradation Evaluation: confirms degraded fact match (10/12) due to split sentences
  4. 7 Agent Scenarios: exact expected statuses and refund amounts
  5. S1 Specifics: exactly 5 model calls and refund INR 128.25 within 0.01
  6. S7 Max Iterations Fallback: exactly 6 calls under default 6000 token limit
  7. Safe-Stopping Fallbacks:
     - S1 with token_limit=200 -> FALLBACK_TOKEN_LIMIT (3 calls, 313 tokens, stops before calculate_refund)
     - S1 with time_limit_seconds=0 -> FALLBACK_TIMEOUT (0 calls, 0 tokens)
  8. Crash / Resume Execution: exact ordered tool sequence
     ['get_order', 'search_policy', 'calculate_refund', 'issue_refund'] with no duplicates
  9. Checkpoint JSON Files & JSONL Outputs: valid schema, valid JSON on every line
 10. Financial Idempotency & Deliverable Preservation: zero duplicate refunds and all Parts A, B, C outputs intact
"""

import os
import sys
import json
import pandas as pd
from collections import Counter

from rag_assistant import chunk_policy_documents
from refund_agent import (
    run_agent,
    resume_agent,
    issue_refund,
    LEDGER_PATH,
    RUNS_PATH,
    STATE_DIR,
)


def validate_chunking():
    print("--- 1. Validating Policy Document Chunking ---")
    chunks = chunk_policy_documents(overlap=10)
    assert len(chunks) == 15, f"Expected exactly 15 chunks, got {len(chunks)}"

    category_counts = Counter(c["category"] for c in chunks)
    expected_counts = {
        "refund": 4,
        "delivery": 3,
        "escalation": 3,
        "payment": 3,
        "privacy": 2,
    }
    assert dict(category_counts) == expected_counts, (
        f"Category chunk counts mismatch: {dict(category_counts)} != {expected_counts}"
    )
    print(f"Checkpoint 1/10 [15 Chunks: {dict(category_counts)}]: PASS")


def validate_overlap10_rag():
    print("--- 2. Validating Overlap-10 RAG Evaluation ---")
    path = os.path.join("outputs", "rag_evaluation_overlap10.csv")
    assert os.path.exists(path), f"Missing {path}"

    df = pd.read_csv(path)
    assert len(df) == 14, f"Expected 14 rows, got {len(df)}"

    in_domain = df[df["expected_doc"] != "NONE"]
    out_domain = df[df["expected_doc"] == "NONE"]

    assert len(in_domain) == 12, f"Expected 12 in-domain questions, got {len(in_domain)}"
    assert len(out_domain) == 2, f"Expected 2 out-of-domain questions, got {len(out_domain)}"

    doc_matches = int(in_domain["doc_match"].astype(bool).sum())
    fact_matches = int(in_domain["fact_match"].astype(bool).sum())
    fallback_matches = int(out_domain["fallback_match"].astype(bool).sum())

    assert doc_matches == 12, f"Doc matches mismatch: {doc_matches}/12"
    assert fact_matches == 12, f"Fact matches mismatch: {fact_matches}/12"
    assert fallback_matches == 2, f"Fallback matches mismatch: {fallback_matches}/2"

    print(f"Checkpoint 2/10 [Overlap-10 RAG: Doc 12/12, Fact 12/12, Fallback 2/2]: PASS")


def validate_overlap0_rag():
    print("--- 3. Validating Zero-Overlap Degradation ---")
    path = os.path.join("outputs", "rag_evaluation_overlap0.csv")
    assert os.path.exists(path), f"Missing {path}"

    df0 = pd.read_csv(path)
    assert len(df0) == 14, f"Expected 14 rows, got {len(df0)}"

    in_domain0 = df0[df0["expected_doc"] != "NONE"]
    fact_matches0 = int(in_domain0["fact_match"].astype(bool).sum())

    # Must demonstrate degradation compared to overlap=10 (10/12 vs 12/12)
    assert fact_matches0 < 12, f"Zero-overlap should show degraded fact matches, but got {fact_matches0}/12"
    assert fact_matches0 == 10, f"Expected exactly 10/12 fact matches for zero-overlap, got {fact_matches0}/12"

    failed_questions = in_domain0[~in_domain0["fact_match"].astype(bool)]["question"].tolist()
    assert len(failed_questions) == 2, f"Expected 2 failed questions, got {len(failed_questions)}"

    print(f"Checkpoint 3/10 [Zero-Overlap Degradation: 10/12 Fact Matches (2 Failed)]: PASS")


def validate_agent_scenarios():
    print("--- 4. Validating 7 Agent Scenarios & Deliverable JSONL ---")
    assert os.path.exists(RUNS_PATH), f"Missing {RUNS_PATH}"

    runs = []
    with open(RUNS_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            assert line, f"Empty line in {RUNS_PATH} at line {idx}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise AssertionError(f"Malformed JSON in {RUNS_PATH} at line {idx}: {e}")
            runs.append(record)

    assert len(runs) == 7, f"Expected 7 scenario runs in {RUNS_PATH}, got {len(runs)}"

    runs_by_id = {r["scenario_id"]: r for r in runs}
    expected_order = [f"S{i}" for i in range(1, 8)]
    assert list(runs_by_id.keys()) == expected_order, f"Scenario IDs mismatch: {list(runs_by_id.keys())}"

    # Expected outcomes
    expected_specs = {
        "S1": {"status": "REFUND", "refund_inr": 128.25, "calls": 5},
        "S2": {"status": "REFUND", "refund_inr": 135.00, "calls": 5},
        "S3": {"status": "ESCALATE", "refund_inr": 1999.00, "calls": 5},
        "S4": {"status": "DENIED", "refund_inr": 0.0, "calls": 4},
        "S5": {"status": "DENIED", "refund_inr": 0.0, "calls": 3},
        "S6": {"status": "NOT_FOUND", "refund_inr": 0.0, "calls": 2},
        "S7": {"status": "FALLBACK_MAX_ITERATIONS", "refund_inr": 0.0, "calls": 6},
    }

    for sid, spec in expected_specs.items():
        run = runs_by_id[sid]
        assert run["status"] == spec["status"], (
            f"Scenario {sid} status mismatch: {run['status']} != {spec['status']}"
        )
        assert abs(run["refund_amount"] - spec["refund_inr"]) <= 0.01, (
            f"Scenario {sid} refund mismatch: {run['refund_amount']} != {spec['refund_inr']}"
        )
        assert run["llm_calls"] == spec["calls"], (
            f"Scenario {sid} calls mismatch: {run['llm_calls']} != {spec['calls']}"
        )

    print("Checkpoint 4/10 [All 7 Scenarios: Statuses and Refund Amounts]: PASS")


def validate_s1_and_s7_specifics():
    print("--- 5. Validating Scenario S1 and S7 Specific Requirements ---")
    runs_by_id = {}
    with open(RUNS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            runs_by_id[r["scenario_id"]] = r

    s1 = runs_by_id["S1"]
    assert s1["llm_calls"] == 5, f"S1 expected exactly 5 model calls, got {s1['llm_calls']}"
    assert abs(s1["refund_amount"] - 128.25) <= 0.01, f"S1 refund mismatch: {s1['refund_amount']}"
    print(f"Checkpoint 5/10 [S1: Exactly 5 Model Calls, Refund INR {s1['refund_amount']}]: PASS")

    s7 = runs_by_id["S7"]
    assert s7["llm_calls"] == 6, f"S7 expected 6 model calls under default budget, got {s7['llm_calls']}"
    assert s7["tokens_used"] <= 6000, f"S7 exceeded default token budget of 6000: {s7['tokens_used']}"
    assert s7["status"] == "FALLBACK_MAX_ITERATIONS", f"S7 expected FALLBACK_MAX_ITERATIONS, got {s7['status']}"
    print(f"Checkpoint 6/10 [S7: 6 Calls, {s7['tokens_used']} Tokens <= 6000, FALLBACK_MAX_ITERATIONS]: PASS")


def validate_safe_stopping_mechanisms():
    print("--- 6. Validating Safe-Stopping Fallbacks (Token Limit & Timeout) ---")
    scenarios_df = pd.read_csv(os.path.join("data", "raw", "agent_scenarios.csv"))
    s1_msg = scenarios_df[scenarios_df["scenario_id"] == "S1"]["message"].values[0]

    # Test A: Token limit fallback (token_limit=200)
    res_tok = run_agent(
        user_message=s1_msg,
        scenario_id="S1_VAL_TOK",
        token_limit=200,
        run_id="val_s1_token_limit",
    )
    assert res_tok["status"] == "FALLBACK_TOKEN_LIMIT", (
        f"Expected FALLBACK_TOKEN_LIMIT, got {res_tok['status']}"
    )
    assert res_tok["llm_calls"] == 3, f"Expected 3 calls for token limit demo, got {res_tok['llm_calls']}"
    assert res_tok["tokens_used"] > 200, f"Tokens used should exceed limit 200, got {res_tok['tokens_used']}"
    assert res_tok["tokens_used"] == 313, f"Expected exactly 313 tokens, got {res_tok['tokens_used']}"
    # Verify tool was not executed after token overrun
    assert res_tok["tools_executed"] == ["get_order", "search_policy"], (
        f"Tools executed before stop mismatch: {res_tok['tools_executed']}"
    )
    assert res_tok["refund_amount"] == 0.0, "No refund should be issued on token limit fallback"

    # Test B: Timeout fallback (time_limit_seconds=0)
    res_time = run_agent(
        user_message=s1_msg,
        scenario_id="S1_VAL_TIME",
        time_limit_seconds=0,
        run_id="val_s1_timeout",
    )
    assert res_time["status"] == "FALLBACK_TIMEOUT", (
        f"Expected FALLBACK_TIMEOUT, got {res_time['status']}"
    )
    assert res_time["llm_calls"] == 0, f"Expected 0 calls for immediate timeout, got {res_time['llm_calls']}"
    assert res_time["tokens_used"] == 0, f"Expected 0 tokens for immediate timeout, got {res_time['tokens_used']}"
    assert res_time["refund_amount"] == 0.0, "No refund should be issued on timeout fallback"

    print("Checkpoint 7/10 [Safe-Stopping: Token Limit (313 tokens) & Timeout (0s)]: PASS")


def validate_crash_resume_and_checkpoints():
    print("--- 7. Validating Crash / Resume Mechanics & State Checkpoints ---")
    scenarios_df = pd.read_csv(os.path.join("data", "raw", "agent_scenarios.csv"))
    s1_msg = scenarios_df[scenarios_df["scenario_id"] == "S1"]["message"].values[0]
    demo_run_id = "s1_resumable_val"

    # Step 1: Run with simulated crash after tool step 2
    try:
        run_agent(
            user_message=s1_msg,
            scenario_id="S1",
            simulate_crash_after_tool_step=2,
            run_id=demo_run_id,
        )
        raise AssertionError("Simulated crash was not triggered!")
    except RuntimeError as e:
        assert "Simulated process crash" in str(e)

    # Step 2: Validate checkpoint JSON file
    ckpt_file = os.path.join(STATE_DIR, f"{demo_run_id}.json")
    assert os.path.exists(ckpt_file), f"Checkpoint file missing: {ckpt_file}"

    with open(ckpt_file, "r", encoding="utf-8") as f:
        ckpt_data = json.load(f)

    required_ckpt_keys = [
        "run_id", "scenario_id", "messages", "tokens_used",
        "llm_calls", "step_number", "tools_executed"
    ]
    for k in required_ckpt_keys:
        assert k in ckpt_data, f"Checkpoint missing key: {k}"

    assert ckpt_data["scenario_id"] == "S1"
    assert ckpt_data["step_number"] == 2
    assert ckpt_data["tools_executed"] == ["get_order", "search_policy"]

    # Step 3: Resume execution
    resumed = resume_agent(demo_run_id)
    assert resumed["status"] == "REFUND", f"Resumed run status expected REFUND, got {resumed['status']}"
    assert abs(resumed["refund_amount"] - 128.25) <= 0.01, f"Resumed refund mismatch: {resumed['refund_amount']}"

    # Step 4: Validate exact ordered tool execution without duplicates
    expected_lifecycle_tools = ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert resumed["tools_executed"] == expected_lifecycle_tools, (
        f"Ordered tools lifecycle mismatch: {resumed['tools_executed']} != {expected_lifecycle_tools}"
    )
    assert len(resumed["tools_executed"]) == len(set(resumed["tools_executed"])), (
        f"Duplicate tool executions found: {resumed['tools_executed']}"
    )

    print("Checkpoint 8/10 [Crash/Resume: Ordered Sequence & Zero Duplicates]: PASS")


def validate_jsonl_outputs_and_idempotency():
    print("--- 8. Validating JSONL Deliverables & Refund Idempotency ---")
    assert os.path.exists(LEDGER_PATH), f"Missing ledger file: {LEDGER_PATH}"

    ledger_records = []
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            assert line, f"Empty line in {LEDGER_PATH} at line {idx}"
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise AssertionError(f"Malformed JSON in {LEDGER_PATH} at line {idx}: {e}")
            for req_key in ["timestamp", "tool", "order_id", "refund_inr"]:
                assert req_key in rec, f"Ledger entry missing key '{req_key}': {rec}"
            ledger_records.append(rec)

    # Check uniqueness of issued refunds
    issued_orders = [r["order_id"] for r in ledger_records if r["tool"] == "issue_refund"]
    assert len(issued_orders) == len(set(issued_orders)), (
        f"Duplicate issued refund order IDs found: {issued_orders}"
    )

    # Live Idempotency Test: re-issuing refund for ORD-100048 must NOT append a duplicate line
    count_before = len(ledger_records)
    issue_refund("ORD-100048", 128.25)

    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        lines_after = [line.strip() for line in f if line.strip()]
    assert len(lines_after) == count_before, (
        f"Idempotency violation! Ledger line count increased from {count_before} to {len(lines_after)}"
    )

    print("Checkpoint 9/10 [JSONL Schema Valid & Idempotency Verified (Zero Duplicates)]: PASS")


def validate_prior_deliverables():
    print("--- 9. Validating Preservation of Parts A, B, and C Deliverables ---")
    prior_deliverables = [
        # Part A
        "data/clean/customers.csv",
        "data/clean/products.csv",
        "data/clean/orders.csv",
        "data/instamart.db",
        "sql/queries.sql",
        # Part B
        "outputs/customer_segments.csv",
        "outputs/lapse_model.joblib",
        "outputs/retention_list.csv",
        # Part C
        "outputs/triage_results.csv",
        "outputs/triage_budget_demo.csv",
    ]

    for rel_path in prior_deliverables:
        assert os.path.exists(rel_path), f"Prior deliverable missing: {rel_path}"
        assert os.path.getsize(rel_path) > 0, f"Prior deliverable is empty: {rel_path}"

    print(f"Checkpoint 10/10 [All 10 Prior Deliverables (Parts A, B, C) Intact]: PASS")


def main():
    print("=" * 75)
    print("PART D: COMPREHENSIVE INDEPENDENT VALIDATION SUITE")
    print("=" * 75)

    try:
        validate_chunking()
        validate_overlap10_rag()
        validate_overlap0_rag()
        validate_agent_scenarios()
        validate_s1_and_s7_specifics()
        validate_safe_stopping_mechanisms()
        validate_crash_resume_and_checkpoints()
        validate_jsonl_outputs_and_idempotency()
        validate_prior_deliverables()

        print("\n" + "=" * 75)
        print("ALL 10 PART D CHECKPOINTS INDEPENDENTLY VALIDATED & PASSED!")
        print("=" * 75)
        return 0
    except AssertionError as err:
        print(f"\n[VALIDATION FAILURE]: {err}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[UNEXPECTED ERROR]: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
