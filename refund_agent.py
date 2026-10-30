"""
refund_agent.py -- Part D Tasks 5-8: Autonomous Refund Agent with ReAct Loop,
Safe-Stopping Mechanisms, Resumable Checkpoints, and Ledger Auditing.

Tasks implemented:
  - Task 5: Tool map (get_order, search_policy, calculate_refund, issue_refund, escalate_to_human),
            TOOLS_SPEC definitions, SQLite order querying, idempotency, and refund ledger logging.
  - Task 6: ReAct agent execution loop (run_agent) for the 7 official scenarios in agent_scenarios.csv,
            logging each run to outputs/agent_runs.jsonl.
  - Task 7: Three safe-stopping mechanisms:
            (a) FALLBACK_MAX_ITERATIONS (max_iterations=6) -> demonstrated by S7
            (b) FALLBACK_TOKEN_LIMIT (token_limit=200)      -> demonstrated by S1
            (c) FALLBACK_TIMEOUT (time_limit_seconds=0)     -> demonstrated by S1
  - Task 8: Resumable checkpoints in state/<run_id>.json, resume_agent(), and crash/resume demonstration.
"""

import json
import os
import re
import sqlite3
import time
import pandas as pd
import mock_llm
from rag_assistant import PolicyRetriever, chunk_policy_documents


DB_PATH = os.path.join("data", "instamart.db")
LEDGER_PATH = os.path.join("outputs", "refund_ledger.jsonl")
RUNS_PATH = os.path.join("outputs", "agent_runs.jsonl")
STATE_DIR = "state"
SCENARIOS_PATH = os.path.join("data", "raw", "agent_scenarios.csv")

# Initialize retriever on standard overlap=10 policy chunks (15 chunks)
_POLICY_CHUNKS = chunk_policy_documents(overlap=10)
_POLICY_RETRIEVER = PolicyRetriever(_POLICY_CHUNKS)


# =============================================================================
# TASK 5: TOOL DEFINITIONS & TOOL MAP
# =============================================================================
def get_order(order_id: str) -> dict:
    """
    Looks up order details in SQLite database data/instamart.db.
    Returns:
      {'order_id': str, 'status': str, 'order_amount': float, 'delivery_minutes': float or None}
      or {'error': 'order not found'}
    """
    assert os.path.exists(DB_PATH), f"Database not found at {DB_PATH}"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
        (order_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": str(row[0]),
        "status": str(row[1]),
        "order_amount": round(float(row[2]), 2),
        "delivery_minutes": None if row[3] is None else round(float(row[3]), 2),
    }


def search_policy(query: str) -> dict:
    """
    Retrieves policy evidence matching query using the TF-IDF PolicyRetriever.
    Returns:
      {'results': list of retrieved chunk dicts}
    """
    results = _POLICY_RETRIEVER.retrieve(query, k=3)
    return {"results": results}


def calculate_refund(order_amount: float, delivery_minutes: float | None) -> dict:
    """
    Calculates refund amount according to policy rules:
      - Completed order delivered in > 45 minutes -> 100% refund of order_amount.
      - Completed order delivered in <= 45 minutes -> 0 refund.
      - Non-delivered / missing delivery minutes -> 0 refund.
    """
    try:
        amt = float(order_amount)
    except (ValueError, TypeError):
        amt = 0.0

    if delivery_minutes is not None:
        try:
            mins = float(delivery_minutes)
            if mins > 45.0:
                return {"refund_inr": round(amt, 2)}
        except (ValueError, TypeError):
            pass

    return {"refund_inr": 0.0}


def issue_refund(order_id: str, refund_inr: float) -> dict:
    """
    Issues refund up to INR 1000 and logs entry to outputs/refund_ledger.jsonl.
    Enforces idempotency: if order_id already has an issued refund in the ledger,
    returns existing record without creating duplicate ledger entries.
    """
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    amt = round(float(refund_inr), 2)

    # Idempotency check: inspect existing ledger for prior issue_refund for order_id
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("order_id") == order_id and entry.get("tool") == "issue_refund":
                        # Already issued; return idempotently without duplicate write
                        return {"refund_inr": float(entry["refund_inr"])}
                except json.JSONDecodeError:
                    continue

    ledger_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": amt,
    }

    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")

    return {"refund_inr": amt}


def escalate_to_human(order_id: str, refund_inr: float) -> dict:
    """
    Escalates refund above INR 1000 to human lead and logs entry to outputs/refund_ledger.jsonl.
    Enforces idempotency to avoid duplicate escalation entries.
    """
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    amt = round(float(refund_inr), 2)

    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("order_id") == order_id and entry.get("tool") == "escalate_to_human":
                        return {"refund_inr": float(entry["refund_inr"])}
                except json.JSONDecodeError:
                    continue

    ledger_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": amt,
    }

    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry) + "\n")

    return {"refund_inr": amt}


# Tool Map and JSON Specification
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up order record in instamart.db by order_id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "Order ID, e.g. ORD-100048"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Retrieve policy document chunks matching a query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query keywords"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate refund amount in INR based on order amount and delivery duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number", "description": "Order spend in INR"},
                    "delivery_minutes": {"type": "number", "description": "Delivery duration in minutes"},
                },
                "required": ["order_amount", "delivery_minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue approved refund up to INR 1000 and record in ledger.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID"},
                    "refund_inr": {"type": "number", "description": "Approved refund amount in INR"},
                },
                "required": ["order_id", "refund_inr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate refund exceeding INR 1000 to human support lead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID"},
                    "refund_inr": {"type": "number", "description": "Escalated refund amount in INR"},
                },
                "required": ["order_id", "refund_inr"],
            },
        },
    },
]


# =============================================================================
# TASK 6 & 7: REACT AGENT LOOP WITH SAFE STOPPING
# =============================================================================
def run_agent(
    user_message: str,
    scenario_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10.0,
    simulate_crash_after_tool_step: int | None = None,
    run_id: str | None = None,
) -> dict:
    """
    Executes the multi-step ReAct agent loop for refund processing.

    Enforces 3 safe-stopping mechanisms:
      1. Timeout check at top of iteration (elapsed >= time_limit_seconds -> FALLBACK_TIMEOUT)
      2. Token limit check after model call (tokens_used > token_limit -> FALLBACK_TOKEN_LIMIT)
      3. Max iterations without answer (loop completion -> FALLBACK_MAX_ITERATIONS)

    Persists checkpoints to state/<run_id>.json after every tool execution.
    """
    os.makedirs(STATE_DIR, exist_ok=True)
    if run_id is None:
        run_id = f"run_{scenario_id}_{int(time.time()*1000)}"

    start_time = time.monotonic()
    messages = [
        {"role": "system", "content": "You are an autonomous support AGENT assisting with customer refunds."},
        {"role": "user", "content": user_message},
    ]

    tokens_used = 0
    llm_calls = 0
    steps = []
    tools_executed = []

    for iteration in range(max_iterations):
        # 1. Timeout Check at Top of Iteration
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK: Execution time limit reached. Handoff to human support.",
                "refund_amount": 0.0,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed,
            }

        # 2. Call mock LLM
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens

        # 3. Token Limit Check (stop immediately without executing requested tool)
        if tokens_used > token_limit:
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK: Token budget limit reached. Handoff to human support.",
                "refund_amount": 0.0,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed,
            }

        msg = response.choices[0].message

        # 4. Final Answer Check (no tool calls requested)
        if not msg.tool_calls:
            final_text = msg.content or ""
            # Normal status is word before colon (REFUND, ESCALATE, DENIED, NOT_FOUND)
            status = final_text.split(":")[0].strip() if ":" in final_text else "DONE"

            # Determine actual refund amount from executed refund tools
            refund_amount = 0.0
            for s in reversed(steps):
                if "refund_inr" in s.get("observation", {}):
                    refund_amount = round(float(s["observation"]["refund_inr"]), 2)
                    break

            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": status,
                "final_answer": final_text,
                "refund_amount": refund_amount,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed,
            }

        # 5. Tool Call Execution
        tc = msg.tool_calls[0]
        tool_name = tc.name
        tool_args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
        call_id = tc.id

        # Append assistant message
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": call_id, "name": tool_name, "arguments": tool_args}],
        })

        # Execute tool function
        tool_fn = TOOL_MAP[tool_name]
        observation = tool_fn(**tool_args)
        tools_executed.append(tool_name)

        # Append tool message with compact JSON content
        obs_json = json.dumps(observation, separators=(",", ":"))
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool_name,
            "content": obs_json,
        })

        step_record = {
            "step": len(steps) + 1,
            "tool": tool_name,
            "arguments": tool_args,
            "observation": observation,
        }
        steps.append(step_record)

        # 6. Save State Checkpoint (Task 8)
        checkpoint_data = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "messages": messages,
            "tokens_used": tokens_used,
            "llm_calls": llm_calls,
            "step_number": len(steps),
            "tools_executed": list(tools_executed),
        }
        with open(os.path.join(STATE_DIR, f"{run_id}.json"), "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)

        # Simulated crash trigger for Task 8 demo
        if simulate_crash_after_tool_step is not None and len(steps) == simulate_crash_after_tool_step:
            raise RuntimeError(f"Simulated process crash immediately after tool step {simulate_crash_after_tool_step}")

    # Fallback when maximum iterations are exhausted without final answer
    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK: Maximum allowed iterations completed without resolution. Handoff to human support.",
        "refund_amount": 0.0,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "tools_executed": tools_executed,
    }


# =============================================================================
# TASK 8: RESUME AGENT FROM CHECKPOINT
# =============================================================================
def resume_agent(
    run_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10.0,
) -> dict:
    """
    Loads saved checkpoint from state/<run_id>.json and resumes execution
    without re-executing already completed tools.
    """
    ckpt_path = os.path.join(STATE_DIR, f"{run_id}.json")
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"

    with open(ckpt_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    scenario_id = checkpoint["scenario_id"]
    messages = checkpoint["messages"]
    tokens_used = checkpoint["tokens_used"]
    llm_calls = checkpoint["llm_calls"]
    tools_executed = list(checkpoint["tools_executed"])
    steps = [
        {"step": i + 1, "tool": t, "arguments": {}, "observation": {}}
        for i, t in enumerate(tools_executed)
    ]

    start_time = time.monotonic()

    for iteration in range(llm_calls, max_iterations):
        # 1. Timeout Check
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK: Execution timed out. Handoff to human support.",
                "refund_amount": 0.0,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed,
            }

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens

        # 2. Token Limit Check
        if tokens_used > token_limit:
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK: Token budget limit reached. Handoff to human support.",
                "refund_amount": 0.0,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed,
            }

        msg = response.choices[0].message

        # 3. Final Answer Check
        if not msg.tool_calls:
            final_text = msg.content or ""
            status = final_text.split(":")[0].strip() if ":" in final_text else "DONE"
            # Determine actual refund amount from executed refund tools
            refund_amount = 0.0
            for s in reversed(steps):
                if "refund_inr" in s.get("observation", {}):
                    refund_amount = round(float(s["observation"]["refund_inr"]), 2)
                    break

            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": status,
                "final_answer": final_text,
                "refund_amount": refund_amount,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed,
            }

        # 4. Tool Execution
        tc = msg.tool_calls[0]
        tool_name = tc.name
        tool_args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
        call_id = tc.id

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": call_id, "name": tool_name, "arguments": tool_args}],
        })

        tool_fn = TOOL_MAP[tool_name]
        observation = tool_fn(**tool_args)
        tools_executed.append(tool_name)

        obs_json = json.dumps(observation, separators=(",", ":"))
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool_name,
            "content": obs_json,
        })

        step_record = {
            "step": len(steps) + 1,
            "tool": tool_name,
            "arguments": tool_args,
            "observation": observation,
        }
        steps.append(step_record)

        # Update checkpoint
        checkpoint_data = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "messages": messages,
            "tokens_used": tokens_used,
            "llm_calls": llm_calls,
            "step_number": len(steps),
            "tools_executed": list(tools_executed),
        }
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK: Maximum iterations completed. Handoff to human support.",
        "refund_amount": 0.0,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "tools_executed": tools_executed,
    }


# =============================================================================
# SCENARIOS EXECUTION & VALIDATION
# =============================================================================
def run_all_agent_scenarios() -> list[dict]:
    """
    Executes all 7 official scenarios from agent_scenarios.csv.
    Logs each scenario run as one valid JSON line to outputs/agent_runs.jsonl.
    """
    print("\n" + "=" * 75)
    print("TASK 6: RUNNING 7 REFUND AGENT SCENARIOS")
    print("=" * 75)

    assert os.path.exists(SCENARIOS_PATH), f"Missing scenarios file: {SCENARIOS_PATH}"
    df_scenarios = pd.read_csv(SCENARIOS_PATH)
    os.makedirs(os.path.dirname(RUNS_PATH), exist_ok=True)

    # Clean runs log and ledger for fresh, reproducible execution
    if os.path.exists(RUNS_PATH):
        os.remove(RUNS_PATH)
    if os.path.exists(LEDGER_PATH):
        os.remove(LEDGER_PATH)

    results = []

    for idx, row in df_scenarios.iterrows():
        sid = row["scenario_id"]
        cid = row["customer_id"]
        msg = row["message"]
        exp_outcome = row["expected_outcome"]
        exp_refund = round(float(row["expected_refund_inr"]), 2)

        run_res = run_agent(msg, scenario_id=sid, max_iterations=6, token_limit=6000)
        results.append(run_res)

        # Append one JSON line per run to outputs/agent_runs.jsonl
        with open(RUNS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_res) + "\n")

    # Display results table
    print("\nSeven Scenario Results Summary Table:")
    print(f"  {'Scenario':<10}  {'Status':<25}  {'Refund (INR)':>12}  {'Calls':>7}  {'Tokens':>8}  {'Expected Status':<18}")
    print(f"  {'-'*10}  {'-'*25}  {'-'*12}  {'-'*7}  {'-'*8}  {'-'*18}")
    for idx, r in enumerate(results):
        exp_row = df_scenarios.iloc[idx]
        exp_status = "FALLBACK_MAX_ITERATIONS" if exp_row["expected_outcome"] == "FALLBACK" else exp_row["expected_outcome"]
        print(f"  {r['scenario_id']:<10}  {r['status']:<25}  {r['refund_amount']:>12.2f}  {r['llm_calls']:>7d}  {r['tokens_used']:>8d}  {exp_status:<18}")

    return results


def run_safe_stopping_demos() -> dict:
    """
    Executes and validates the 3 safe-stopping demonstrations:
      1. S7 with default token_limit=6000 -> FALLBACK_MAX_ITERATIONS (6 calls)
      2. S1 with token_limit=200          -> FALLBACK_TOKEN_LIMIT
      3. S1 with time_limit_seconds=0     -> FALLBACK_TIMEOUT
    """
    print("\n" + "=" * 75)
    print("TASK 7: SAFE-STOPPING DEMONSTRATIONS")
    print("=" * 75)

    df_scenarios = pd.read_csv(SCENARIOS_PATH)
    s1_msg = df_scenarios[df_scenarios["scenario_id"] == "S1"]["message"].values[0]
    s7_msg = df_scenarios[df_scenarios["scenario_id"] == "S7"]["message"].values[0]

    # Demo A: S7 Max Iterations (default token_limit=6000)
    res_max_iter = run_agent(s7_msg, scenario_id="S7_max_iter", max_iterations=6, token_limit=6000)
    print(f"  Demo A [Max Iterations] S7: status = {res_max_iter['status']} (Calls: {res_max_iter['llm_calls']}, Tokens: {res_max_iter['tokens_used']})")
    assert res_max_iter["status"] == "FALLBACK_MAX_ITERATIONS"
    assert res_max_iter["llm_calls"] == 6

    # Demo B: S1 Token Limit (token_limit=200)
    res_token_limit = run_agent(s1_msg, scenario_id="S1_token_limit", token_limit=200)
    print(f"  Demo B [Token Limit]    S1: status = {res_token_limit['status']} (Calls: {res_token_limit['llm_calls']}, Tokens: {res_token_limit['tokens_used']})")
    assert res_token_limit["status"] == "FALLBACK_TOKEN_LIMIT"

    # Demo C: S1 Timeout (time_limit_seconds=0.0)
    res_timeout = run_agent(s1_msg, scenario_id="S1_timeout", time_limit_seconds=0.0)
    print(f"  Demo C [Timeout]        S1: status = {res_timeout['status']} (Calls: {res_timeout['llm_calls']}, Tokens: {res_timeout['tokens_used']})")
    assert res_timeout["status"] == "FALLBACK_TIMEOUT"

    print(">> Task 7 Checkpoint [All 3 Safe-Stopping Fallbacks Successfully Demonstrated]: PASS")
    return {
        "s7_max_iter": res_max_iter,
        "s1_token_limit": res_token_limit,
        "s1_timeout": res_timeout,
    }


def run_crash_resume_demo() -> dict:
    """
    Demonstrates resumable checkpointing on Scenario S1:
      1. Starts S1.
      2. Executes get_order and search_policy.
      3. Saves checkpoint after step 2.
      4. Simulates process crash.
      5. Calls resume_agent(run_id) and finishes with calculate_refund and issue_refund.
      6. Verifies exact tool sequence and final refund amount.
    """
    print("\n" + "=" * 75)
    print("TASK 8: CRASH AND RESUME DEMONSTRATION ON SCENARIO S1")
    print("=" * 75)

    df_scenarios = pd.read_csv(SCENARIOS_PATH)
    s1_msg = df_scenarios[df_scenarios["scenario_id"] == "S1"]["message"].values[0]
    crash_run_id = "s1_resumable_demo"

    # Clean old demo checkpoint if present
    demo_ckpt_path = os.path.join(STATE_DIR, f"{crash_run_id}.json")
    if os.path.exists(demo_ckpt_path):
        os.remove(demo_ckpt_path)

    # 1. Execute up to step 2 then simulate crash
    crash_occurred = False
    try:
        run_agent(
            user_message=s1_msg,
            scenario_id="S1",
            simulate_crash_after_tool_step=2,
            run_id=crash_run_id,
        )
    except RuntimeError as err:
        crash_occurred = True
        print(f"  Step 1: Simulated crash triggered as expected: {err}")

    assert crash_occurred, "Simulated crash did not trigger!"
    assert os.path.exists(demo_ckpt_path), f"Checkpoint was not saved at {demo_ckpt_path}"

    with open(demo_ckpt_path, "r", encoding="utf-8") as f:
        pre_crash_ckpt = json.load(f)
    print(f"  Step 2: Verified pre-crash checkpoint saved with {pre_crash_ckpt['step_number']} tools: {pre_crash_ckpt['tools_executed']}")
    assert pre_crash_ckpt["tools_executed"] == ["get_order", "search_policy"]

    # 2. Call resume_agent
    print("  Step 3: Calling resume_agent to recover from checkpoint...")
    resumed_res = resume_agent(crash_run_id)

    print(f"  Step 4: Resumed run completed with status: {resumed_res['status']}")
    print(f"          Final Answer: {resumed_res['final_answer']}")
    print(f"          Refund Amount: INR {resumed_res['refund_amount']:.2f}")
    print(f"          Ordered Tools Executed Across Run: {resumed_res['tools_executed']}")

    # Checkpoint validations
    expected_tools = ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert resumed_res["tools_executed"] == expected_tools, (
        f"Tools sequence mismatch: {resumed_res['tools_executed']} != {expected_tools}"
    )
    assert resumed_res["status"] == "REFUND", f"Expected REFUND, got {resumed_res['status']}"
    assert abs(resumed_res["refund_amount"] - 128.25) <= 0.01, f"Expected 128.25, got {resumed_res['refund_amount']}"

    print(f">> Task 8 Checkpoint [Crash/Resume Sequence: {' -> '.join(expected_tools)}]: PASS")
    return resumed_res


# =============================================================================
# PART D COMPREHENSIVE VALIDATION SUITE
# =============================================================================
def validate_part_d(scenario_results: list[dict]):
    """
    Performs comprehensive verification of Part D acceptance criteria.
    """
    print("\n" + "=" * 75)
    print("PART D: ACCEPTANCE CRITERIA VALIDATION SUITE")
    print("=" * 75)

    # 1. Seven Scenarios Assertions
    df_scenarios = pd.read_csv(SCENARIOS_PATH)
    for idx, r in enumerate(scenario_results):
        exp_row = df_scenarios.iloc[idx]
        sid = exp_row["scenario_id"]
        exp_outcome = exp_row["expected_outcome"]
        exp_refund = round(float(exp_row["expected_refund_inr"]), 2)

        if sid == "S1":
            assert r["status"] == "REFUND", f"S1 expected REFUND, got {r['status']}"
            assert abs(r["refund_amount"] - exp_refund) <= 0.01, f"S1 refund mismatch: {r['refund_amount']} != {exp_refund}"
            assert r["llm_calls"] == 5, f"S1 expected exactly 5 calls, got {r['llm_calls']}"
        elif sid == "S2":
            assert r["status"] == "REFUND", f"S2 expected REFUND, got {r['status']}"
            assert abs(r["refund_amount"] - exp_refund) <= 0.01, f"S2 refund mismatch: {r['refund_amount']} != {exp_refund}"
            assert r["llm_calls"] == 5, f"S2 expected 5 calls, got {r['llm_calls']}"
        elif sid == "S3":
            assert r["status"] == "ESCALATE", f"S3 expected ESCALATE, got {r['status']}"
            assert abs(r["refund_amount"] - exp_refund) <= 0.01, f"S3 refund mismatch: {r['refund_amount']} != {exp_refund}"
            assert r["llm_calls"] == 5, f"S3 expected 5 calls, got {r['llm_calls']}"
        elif sid in ("S4", "S5"):
            assert r["status"] == "DENIED", f"{sid} expected DENIED, got {r['status']}"
            assert r["refund_amount"] == 0.0
            if sid == "S4":
                assert r["llm_calls"] == 4, f"S4 expected 4 calls, got {r['llm_calls']}"
            else:
                assert r["llm_calls"] == 3, f"S5 expected 3 calls, got {r['llm_calls']}"
        elif sid == "S6":
            assert r["status"] == "NOT_FOUND", f"S6 expected NOT_FOUND, got {r['status']}"
            assert r["refund_amount"] == 0.0
            assert r["llm_calls"] == 2, f"S6 expected 2 calls, got {r['llm_calls']}"
        elif sid == "S7":
            assert r["status"] == "FALLBACK_MAX_ITERATIONS", f"S7 expected FALLBACK_MAX_ITERATIONS, got {r['status']}"
            assert r["llm_calls"] == 6, f"S7 expected 6 calls under default limit, got {r['llm_calls']}"
            assert r["tokens_used"] <= 6000, f"S7 exceeded default token budget of 6000: {r['tokens_used']}"

    print(">> Validation 1/5 [All 7 Scenarios Matched Expected Outcomes, S1=5 calls, S7=6 calls under budget]: PASS")

    # 2. Output Files & JSONL Format Check
    assert os.path.exists(RUNS_PATH), f"Missing {RUNS_PATH}"
    assert os.path.exists(LEDGER_PATH), f"Missing {LEDGER_PATH}"
    assert os.path.exists(STATE_DIR), f"Missing state directory {STATE_DIR}"

    runs_count = 0
    with open(RUNS_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            assert line, f"Empty line in {RUNS_PATH} at line {idx}"
            json.loads(line)
            runs_count += 1
    assert runs_count == 7, f"Expected 7 lines in {RUNS_PATH}, got {runs_count}"

    ledger_count = 0
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            assert line, f"Empty line in {LEDGER_PATH} at line {idx}"
            rec = json.loads(line)
            for req_key in ["timestamp", "tool", "order_id", "refund_inr"]:
                assert req_key in rec, f"Missing key {req_key} in ledger entry: {rec}"
            ledger_count += 1
    assert ledger_count >= 1, "Expected at least 1 record in refund ledger"
    print(f">> Validation 2/5 [Valid JSONL Deliverables in outputs/ (agent_runs.jsonl, refund_ledger.jsonl)]: PASS")

    # 3. Checkpoint Files Validation
    state_files = [f for f in os.listdir(STATE_DIR) if f.endswith(".json")]
    assert len(state_files) > 0, "No checkpoint JSON files found in state/"
    for sf in state_files:
        sf_path = os.path.join(STATE_DIR, sf)
        with open(sf_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
            for k in ["run_id", "scenario_id", "messages", "tokens_used", "llm_calls", "step_number", "tools_executed"]:
                assert k in ckpt, f"Checkpoint {sf} missing key {k}"
    print(f">> Validation 3/5 [State Checkpoint JSON Files Validated in {STATE_DIR}]: PASS")

    # 4. Idempotency Check (Confirm no duplicate refund records for same order)
    ledger_orders = []
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("tool") == "issue_refund":
                ledger_orders.append(data.get("order_id"))
    assert len(ledger_orders) == len(set(ledger_orders)), "Duplicate refunds detected in refund ledger!"
    print(">> Validation 4/5 [Idempotency Verified: Zero Duplicate Refunds in Ledger]: PASS")

    # 5. Check that Parts A, B, and C Deliverables are Preserved
    required_prior_files = [
        "data/clean/customers.csv", "data/clean/products.csv", "data/clean/orders.csv",
        "data/instamart.db", "sql/queries.sql",
        "outputs/customer_segments.csv", "outputs/lapse_model.joblib", "outputs/retention_list.csv",
        "outputs/triage_results.csv", "outputs/triage_budget_demo.csv",
    ]
    for p in required_prior_files:
        assert os.path.exists(p), f"Prior deliverable missing: {p}"
        assert os.path.getsize(p) > 0, f"Prior deliverable empty: {p}"
    print(">> Validation 5/5 [All Parts A, B, and C Output Deliverables Preserved]: PASS")

    print("\n" + "=" * 75)
    print("ALL PART D ACCEPTANCE CRITERIA VALIDATED PERFECTLY!")
    print("=" * 75)


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================
def main():
    print("=" * 75)
    print("PART D: AUTONOMOUS REFUND AGENT & REACT WORKFLOW")
    print("=" * 75)

    # 1. Run 7 Official Agent Scenarios
    scenario_results = run_all_agent_scenarios()

    # 2. Run Safe-Stopping Demonstrations
    stopping_demos = run_safe_stopping_demos()

    # 3. Run Crash and Resume Demonstration
    crash_resume_demo = run_crash_resume_demo()

    # 4. Run Part D Validation Suite
    validate_part_d(scenario_results)


if __name__ == "__main__":
    main()
