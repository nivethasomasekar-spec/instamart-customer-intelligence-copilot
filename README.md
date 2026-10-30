# Overview

Instamart Customer Intelligence & Support Copilot is an end-to-end analytical and AI system built across four connected parts:
- **Part A:** Data Foundation: Clean, Store, Query (ETL pipeline, SQLite database, and SQL/pandas analytics)
- **Part B:** Customer Segments and Lapse Prediction (RFM feature engineering, K-Means clustering, and supervised lapse modeling)
- **Part C:** Support Ticket Triage with Guardrails (PII redaction, LLM classification, and segment-aware routing)
- **Part D:** Policy RAG Assistant and Refund Agent (TF-IDF retrieval, grounded QA, and an autonomous ReAct refund agent)

# Reproduce

Numbered shell commands to set up the environment and reproduce the project step by step:

```bash
# 1. Create and activate an isolated virtual environment (Python 3.11+)
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate raw synthetic datasets and policy documents
python generate_data.py

# 4. Part A: Audit raw data, clean tables, store SQLite DB, and run validations
python audit_raw.py
python clean_data.py
python validate_part_a.py
python run_sql_analysis.py
python eda_analysis.py

# 5. Part B: Customer segmentation and lapse prediction
python segmentation.py
python lapse_prediction.py

# 6. Part C: Support ticket triage with guardrails and routing
python triage.py

# 7. Part D: Policy RAG assistant and autonomous refund agent
python rag_assistant.py
python refund_agent.py
python validate_part_d.py
```

# Part A

### Data Foundation (Tasks 1–7)

#### Task 1: Project Setup
Raw CSV datasets (`customers.csv`, `products.csv`, `orders.csv`) and policy documents were generated using `generate_data.py` and stored under `data/raw/`. A `requirements.txt` file pins all permitted dependencies (pandas, numpy, scikit-learn, joblib).

#### Task 2: Raw Data Audit (`audit_raw.py`)
All three raw tables were inspected for shape, data types, missing values, and data quality issues. The 12 official audit reference metrics were verified and all matched exactly:

| Audit Metric | Found | Reference |
|---|:---:|:---:|
| Raw `orders.csv` rows | 13,084 | 13,084 |
| Exact duplicate rows | 119 | 119 |
| Pending-status rows (to remove) | 656 | 656 |
| Distinct `order_id` values | 12,309 | 12,309 |
| Non-numeric `order_amount` entries | 120 | 120 |
| Negative `delivery_minutes` entries | 106 | 106 |
| Blank `delivery_minutes` values | 2,541 | 2,541 |
| Customers with blank age | 61 | 61 |
| Customers with age outside 10–100 | 6 | 6 |
| Distinct raw city spellings | 24 | 24 |
| Distinct non-blank gender spellings | 10 | 10 |
| Customers with blank gender | 30 | 30 |

**Findings:** The raw orders table had 775 rows removed (119 duplicate rows + 656 pending-status rows) and 2,767 corrupt values fixed (120 non-numeric order amounts, 106 negative delivery times, and 2,541 missing delivery minutes requiring imputation). The customers table had 91 records requiring age or gender correction (61 blank ages, 6 out-of-range ages, 30 blank genders).

#### Task 3: Customer Data Cleaning (`clean_data.py`)
Cleaning rules applied to `customers.csv`:
- **City standardisation:** Strip whitespace and apply title case to collapse 24 raw spellings into 6 canonical city names (Pune: 227, Delhi: 210, Bengaluru: 197, Chennai: 191, Hyderabad: 191, Mumbai: 184).
- **Gender standardisation:** Map all variant spellings (M, Male, m, F, Female, f, etc.) to `male`, `female`, or `unknown` (female: 610, male: 560, unknown: 30).
- **Age cleaning:** Treat ages outside the range [10, 100] as missing; impute all 67 missing ages using the median age of the customer's own city group.
- **Result:** 1,200 customers, 5 columns, zero missing values. All ages within [18, 60].

#### Task 4: Order Data Cleaning (`clean_data.py`)
Cleaning rules applied to `orders.csv` in strict sequence:
1. **Timestamp parsing:** Convert `order_ts` to datetime.
2. **Duplicate removal:** Drop 119 exact duplicate rows.
3. **Order-ID deduplication:** Keep only the latest record per `order_id` (removed 656 pending-status rows), leaving 12,309 unique orders.
4. **Order amount repair:** Convert `order_amount` to numeric; impute 120 missing/corrupted values using `price_inr × quantity × (1 − discount_pct / 100)`.
5. **Delivery time repair:** Set 106 negative `delivery_minutes` to missing; impute missing values for **completed** orders using the product-category-level median; leave cancelled orders as missing by design.
- **Result:** 12,309 orders (11,110 completed, 1,199 cancelled, 0 pending), zero missing `order_amount`, zero missing `delivery_minutes` for completed orders. Total completed revenue: **INR 2,998,668.80**.

#### Task 5: Validation (`validate_part_a.py`)
All 8 official reference checkpoints verified and passed:
- Cleaned shapes: Customers (1,200 × 5), Products (40 × 4), Orders (12,309 × 9).
- Completed revenue sum: INR 2,998,668.80 (exact match).
- SQLite tables (`customers`, `products`, `orders`) row counts: 1,200 / 40 / 12,309.
- Timestamp format in SQLite: `YYYY-MM-DD HH:MM:SS` (19-character string).
- **Output:** `ALL CHECKPOINTS PASSED`

#### Task 6: SQL Analysis (`run_sql_analysis.py`, `sql/queries.sql`)
Seven SQL queries executed against `data/instamart.db` using `pandas.read_sql_query`. All reference checkpoints passed:

| Query | Description | Key Result |
|:---:|---|---|
| Q1 | WHERE + HAVING filter | 30 customers with high-value basket spend > INR 8,000 |
| Q2 | 3-Table JOIN (City × Category Revenue) | 30 rows; total INR 2,998,668.80; max cell: Bengaluru / Electronics INR 233,909.45 |
| Q3 | Scalar subquery (above-average product price) | 11 products priced above catalogue average |
| Q4 | LEFT JOIN (customers with no completed orders) | 25 customers |
| Q5 | CTE (above-average total spenders) | 550 customers; average spend INR 2,552.06 |
| Q6 | Window functions RANK / DENSE_RANK / ROW_NUMBER | City leaderboards verified; 8 rows where RANK ≠ ROW_NUMBER (ties) |
| Q7 | Running total (daily revenue) | 183 daily rows; final cumulative revenue INR 2,998,668.80 |

#### Task 7: Pandas EDA (`eda_analysis.py`)
- **Q2 Cross-check:** `pivot_table()` grand total matched SQL grand total exactly (INR 2,998,668.80).
- **City-level aggregations verified:** Bengaluru (sum INR 523,643.40, mean INR 276.47, count 1,894 orders); Mumbai (sum INR 479,058.50, mean INR 297.00, count 1,613 orders).
- **Order amount distribution:** Mean INR 269.91, Median INR 135.00, Skewness 4.61 (heavy right skew). Q1 = INR 64.80, Q3 = INR 297.00, IQR = INR 232.20. Upper 1.5×IQR fence = INR 645.30. Outliers above fence: **897 orders** (mainly high-value electronics); outliers below: 0.
- **Correlation analysis:** Strongest linear correlation is `order_amount` vs. `discount_pct` (r ≈ −0.095); second is `order_amount` vs. `quantity` (r ≈ +0.087). All correlations are negligible in magnitude.

# Part B

### Customer Segmentation (Tasks 1-3)
- **RFM Cohort:** 1,175 customers with at least one completed order as of snapshot 2025-10-01 00:00:00.
  - Recency: Median 19 days, Min 0, Max 180 days.
  - Frequency: Mean 9.46 orders, Max 33 orders.
  - Monetary: Mean INR 2,552.06, Max INR 14,766.55.
- **K-Means Clustering:** Evaluated K=2 through 8; K=4 was chosen based on the elbow method, peak silhouette score (0.526), and business interpretability.
- **PCA Analysis:** 2 components capture 85.2% of total RFM variance (PC1: 63.4%, PC2: 21.8%).
- **Customer Segments (K=4):**
  - **Frequent Daily Shoppers (419 customers):** Mean Recency 8.77 days, Mean Frequency 19.40 orders, Mean Spend INR 3,410.02. Action: Promote daily essentials subscription streaks and category cross-selling coupons.
  - **High-Value/Premium Buyers (138 customers, `is_premium=True`):** Mean Recency 24.12 days, Mean Frequency 6.77 orders, Mean Spend INR 7,129.70. Action: Assign VIP support desk routing and early access to high-margin electronics.
  - **Occasional Grocery Buyers (436 customers):** Mean Recency 28.86 days, Mean Frequency 3.77 orders, Mean Spend INR 1,057.62. Action: Send weekend promotional bundles and minimum-order free delivery nudges.
  - **At-Risk/Dormant Shoppers (182 customers):** Mean Recency 109.04 days, Mean Frequency 2.20 orders, Mean Spend INR 685.99. Action: Trigger targeted win-back notifications with steep reactivation discounts.

### Customer Lapse Prediction (Tasks 4-8)
- **Cutoff Date & Cohort:** 2025-09-01 00:00:00. 1,156 eligible customers who placed at least one completed order before the cutoff.
- **Target Label (`lapsed`):**
  - 1 = Lapsed (zero completed orders in September 2025): 443 customers (38.32%).
  - 0 = Active (at least one completed order in September 2025): 713 customers (61.68%).
- **Data Leakage Guarantees:**
  - No data from 2025-09-01 onwards entered any feature calculation.
  - Customer segment labels from Task 3 were strictly excluded as model features.
  - No encoder, scaler, or resampler saw test rows; preprocessing pipelines were fit exclusively on training data and cross-validation training folds.
- **Stratified Train/Test Split (80/20):**
  - Training set: 924 customers (570 active, 354 lapsed).
  - Test set: 232 customers (143 active, 89 lapsed).
  - Naive baseline (predict always active): Accuracy 61.6%, Lapsed-class F1 = 0.0.
- **Model Tuning (5-fold Cross-Validation, scoring='f1'):**
  - **Logistic Regression Pipeline:** Test F1 = 0.6222, Precision = 0.6154, Recall = 0.6292.
  - **Random Forest Pipeline (Selected Model):** Test F1 = 0.6257, Precision = 0.6222, Recall = 0.6292.
- **Class-Balance Experiment (Task 7):**
  - Training set oversampled with replacement from 354 lapsed rows to 570 lapsed rows (total balanced train set: 1,140 rows; 570 active, 570 lapsed). Test set left completely untouched (232 rows).
  - **Recall Jump:** Logistic Regression recall jumped from 62.9% to 86.5% (identifying 77 of 89 lapsed customers) while precision remained near 60% (59.2%). Random Forest recall jumped from 62.9% to 85.4% (identifying 76 of 89 lapsed customers).
  - **Business Trade-off:** High recall enables capturing significantly more churners before they leave the platform, heavily outweighing the modest marketing cost of false alarm outreach.
- **Model Interpretation & Persistence (Task 8):**
  - **Top 5 Random Forest Features:** `frequency` (29.7%), `monetary` (28.3%), `recency` (12.4%), `cancellation_rate` (5.5%), `avg_delivery_minutes` (5.3%).
  - **Logistic Regression Coefficients:** `frequency` (-1.6266, negative -- higher frequency strongly decreases lapse odds), `favourite_category_Electronics` (-0.9228, negative -- high platform stickiness), `recency` (+0.0425, positive -- longer gap since last order increases lapse odds).
  - **Model Persistence:** Saved to `outputs/lapse_model.joblib`; reload verification confirmed 100% exact match on test predictions.
  - **Retention List:** Saved top 10 at-risk Premium customers to `outputs/retention_list.csv` with columns `customer_id, lapse_probability, actually_lapsed`.

# Part C

### Support Ticket Triage with Guardrails (Tasks 1-7)

- **Total Ingested Tickets:** 164 support tickets from `data/raw/tickets.csv`.
- **Task 1: Input Guardrails:**
  - **Empty Tickets (`skipped_empty`):** Exactly 4 tickets had empty or whitespace-only bodies; they were bypassed before invoking the model and flagged for human review.
  - **Overlength Tickets (`flagged_long`):** Exactly 4 tickets exceeded the threshold of `MAX_INPUT_TOKENS = 400` (each estimated at 734 tokens). These were intercepted to protect token limits and prevent prompt injection or truncation.
  - **Eligible Tickets Processed by Model:** 156 tickets (164 total - 4 empty - 4 long).
- **Task 2: PII Redaction & Privacy Guardrails:**
  - Applied three ordered regular expressions before any ticket text was passed to the LLM:
    1. Email addresses (`[\w.+-]+@[\w-]+\.[\w.]+` → `[EMAIL]`): **55** replacements.
    2. Phone numbers (`\b[6-9]\d{9}\b` → `[PHONE]`): **91** replacements.
    3. Order identifiers (`ORD-\d{6}` → `[ORDER_ID]`): **132** replacements.
  - **Total Redactions:** Exactly **278** substitutions across 164 tickets.
  - **Zero Residual Leakage:** Post-redaction automated scan confirmed **0** unmasked emails, phone numbers, or order IDs remained in the prompt text.
- **Task 3: Prompt Engineering:**
  - **Zero-Shot Prompt:** Enforced the `CLASSIFY` system trigger, listed the four allowed categories (`billing`, `delivery`, `technical`, `fraud`), and required JSON output `{"category": "...", "reason": "..."}` without `EXAMPLES:`.
  - **Few-Shot Prompt:** Included the identical JSON schema requirement plus an explicit `EXAMPLES:` block containing labelled examples (including fraud and technical).
- **Task 4: Resilient API Calls & Exponential Backoff:**
  - Wrapped model calls in a retry handler with a maximum of 3 retries, base delay `BASE_DELAY = 0.05s`, and exponential backoff with jitter (`0.05 * 2^n * uniform(0.9, 1.15)`).
  - Out of 156 eligible tickets, exactly **19 transient 503 errors** were raised by `mock_llm.py`; **100% (19/19) were successfully recovered** on retry 0 with zero unhandled exceptions.
- **Task 5: Output Validation & Fallback Parser:**
  - **`classified_json`:** **135** tickets returned valid JSON with an allowed category.
  - **`classified_keyword`:** **12** tickets returned chatty non-JSON text (`The customer appears to have a <cat>-related problem.`); recovered via regex word boundary parsing (`re.findall(r"[a-z]+", text.lower())`).
  - **`flagged_invalid_output`:** **9** tickets returned categories outside the allowed list (`shipping`); automatically flagged and safely routed to human review without guessing.
- **Task 6: Zero-Shot vs. Few-Shot Comparison:**

| Metric | Zero-Shot Prompt | Few-Shot Prompt | Delta / Advantage |
|---|:---:|:---:|:---:|
| Eligible Tickets Processed | 156 | 156 | -- |
| Tickets Classified | 147 | 147 | Rule-validated |
| **Classification Accuracy** | **85.7%** (126 / 147) | **100.0%** (147 / 147) | **+14.3% accuracy gain** |
| Total Tokens Used | 12,208 | 20,320 | +8,112 tokens (+66.4%) |
| Tokens per Classified Ticket | 83.0 | 138.2 | +55.2 tokens/ticket |

  - **Zero-Shot Error Analysis:** In the zero-shot run, 21 tickets were misclassified (11 fraud tickets mispredicted as billing, and 10 technical tickets mispredicted as billing) due to keyword overlap with payment terms.
  - **Token Economics Explanation:** Few-shot prompt requires an extra ~55 tokens per ticket, representing a marginal cost of approximately $0.00005 per ticket. Because misrouting a fraud ticket to billing delays emergency account freezes and security escalation, the additional token expenditure is overwhelmingly justified by the elimination of catastrophic fraud routing errors.
- **Task 7: Budget Enforcement & Customer-Segment Routing:**
  - **Token Budget Demo (`TOKEN_BUDGET = 6000`):** Saved to `outputs/triage_budget_demo.csv`. The predictive budget check halted execution at **5,962 tokens** (strictly ≤ 6,000 budget), marking **80 tickets** as `budget_exhausted` and routing them to human review.
  - **Production Triage (`outputs/triage_results.csv`):** 164 total tickets processed via Few-Shot prompt with `TOKEN_BUDGET = 100000`. Routing rules applied using Part B customer segments:
    - **`support_queue` (Priority: NORMAL):** 102 tickets (standard non-fraud, non-premium issues).
    - **`security_team` (Priority: HIGH):** 34 tickets (all confirmed fraud cases).
    - **`human_review` (Priority: NORMAL):** 17 tickets (4 empty, 4 long, 9 invalid output).
    - **`premium_desk` (Priority: HIGH):** 11 tickets (tickets from verified Premium customers from Part B).

# Part D

### Policy RAG Assistant (Tasks 1-4)

- **Policy Document Corpus:** 5 store policy documents located in `data/policies/*.txt` (`delivery_policy.txt`, `escalation_policy.txt`, `payment_policy.txt`, `privacy_policy.txt`, `refund_policy.txt`).
- **Sliding-Window Chunking (Task 1):**
  - Chunk size: **40 words**, Overlap: **10 words**, Step: **30 words** (`start += chunk_size - overlap`).
  - Total chunks generated: **15 chunks** across all 5 policy categories.
  - Chunk distribution:
    - `refund_policy.txt`: **4 chunks**
    - `delivery_policy.txt`: **3 chunks**
    - `escalation_policy.txt`: **3 chunks**
    - `payment_policy.txt`: **3 chunks**
    - `privacy_policy.txt`: **2 chunks**
- **TF-IDF Vector Retrieval (Task 2):**
  - Fitted `TfidfVectorizer(stop_words='english')` across the 15 chunk texts.
  - Calculated cosine similarity against input query vectors: `cosine_similarity(query_vec, chunk_matrix)[0]`.
  - Benchmark query `"refund policy late delivery"` returned top 3 chunks, all belonging to `refund_policy.txt` (Scores: 0.2719, 0.2514, 0.2454).
  - **Information Retrieval Concepts:**
    - **Term Frequency (TF):** Measures how often a term occurs within a specific chunk; frequent terms within a chunk indicate topical relevance.
    - **Inverse Document Frequency (IDF):** Penalizes terms that appear ubiquitously across many documents (e.g., "policy", "order"), giving higher weight to distinctive keywords (e.g., "unauthorized", "deducted").
    - **Vocabulary Mismatch Limitation:** Exact n-gram/TF-IDF matching fails when queries use synonyms, paraphrases, or alternative phrasings not present verbatim in the indexed text (e.g., "reimbursement" vs. "refund"), unlike dense semantic embeddings.
- **Grounded Policy QA (Task 3):**
  - Formatted prompt with system trigger `POLICY ASSISTANT` and structured context `CONTEXT:\n{context_text}\n\nQUESTION:\n{question}`.
  - Answer extraction in `mock_llm.py` extracts sentences containing overlapping query keywords or falls back to standard out-of-domain safe rejection: `"I could not find that information in the policy documents."`.
- **RAG Evaluation (Task 4):**
  - Evaluated on all 14 questions in `data/raw/policy_eval.csv` (12 answerable in-domain questions, 2 out-of-domain questions).
  - Saved detailed evaluation to `outputs/rag_evaluation_overlap10.csv`.

| Question | Expected Source | Expected Fact | Top-3 Retrieved Sources | Doc Match | Fact Match | Fallback Match |
|---|---|---|---|:---:|:---:|:---:|
| What delivery time makes a completed order eligible for a full refund? | `refund_policy.txt` | more than 45 minutes | refund, refund, privacy | True | True | False |
| Who must approve a refund above INR 1000? | `refund_policy.txt` | human support lead | refund, refund, refund | True | True | False |
| How many days do I have to raise a refund request? | `refund_policy.txt` | within 7 days | refund, privacy, refund | True | True | False |
| What is the delivery fee on small orders? | `delivery_policy.txt` | INR 25 | delivery, delivery, payment | True | True | False |
| On orders of what value is delivery free? | `delivery_policy.txt` | INR 199 or more | delivery, delivery, payment | True | True | False |
| Why does the electronics delivery need a one time password? | `delivery_policy.txt` | one time password | delivery, delivery, payment | True | True | False |
| Is cash on delivery available for a large order? | `payment_policy.txt` | INR 2500 | payment, payment, payment | True | True | False |
| How soon is money returned when a payment fails but is deducted? | `payment_policy.txt` | 48 hours | payment, payment, refund | True | True | False |
| Can wallet balance be refunded as cash? | `payment_policy.txt` | never refunded as cash | payment, payment, payment | True | True | False |
| When are delivery addresses deleted from active systems? | `privacy_policy.txt` | 180 days | privacy, privacy, delivery | True | True | False |
| How fast must a fraud ticket reach the security team? | `escalation_policy.txt` | within 2 hours | escalation, escalation, escalation | True | True | False |
| After how many steps must an automated assistant hand over to a human? | `escalation_policy.txt` | 6 steps | escalation, escalation, delivery | True | True | False |
| What is the favourite colour of the company mascot? | `NONE` | (Out of Domain) | delivery, refund, delivery | True | False | True |
| Who won the cricket world cup final? | `NONE` | (Out of Domain) | payment, delivery, privacy | True | False | True |

- **Chunking Overlap Comparison (Overlap 10 vs. Overlap 0):**
  - **Overlap 10 Chunks:** 15 total chunks. Answerable Doc Match: **12/12 (100%)**, Answerable Fact Match: **12/12 (100%)**, Fallback Match: **2/2 (100%)**.
  - **Overlap 0 Chunks:** 13 total chunks. Answerable Doc Match: **12/12 (100%)**, Answerable Fact Match: **10/12 (83.3%)**, Fallback Match: **2/2 (100%)**.
  - **Failure Explanation:** Under zero overlap, contiguous sentences are split across arbitrary 40-word boundaries:
    1. *"What delivery time makes a completed order eligible for a full refund?"* failed because `"more than 45 minutes"` was cleaved across the boundary, causing the model to retrieve only `"receives a full refund of the order amount."`.
    2. *"How soon is money returned when a payment fails but is deducted?"* failed because the `"within 48 hours"` resolution window was severed from the opening conditional clause. Sliding overlap preserves semantic continuity across chunk transitions.

---

### Autonomous Refund Agent (Tasks 5-8)

- **Agent Tool Suite & Schemas (Task 5):**
  - Implemented 5 deterministic tools with strict parameter schemas:
    1. `get_order(order_id)`: Queries SQLite database `data/instamart.db` for order status, delivery minutes, total amount, and customer ID.
    2. `search_policy(query)`: Reuses `PolicyRetriever` to return top matching policy chunks.
    3. `calculate_refund(order_id, reason)`: Computes refund eligibility and monetary amount in INR (`late_delivery` > 45 mins gets 100% refund).
    4. `issue_refund(order_id, amount_inr, reason)`: Idempotently credits customer account and appends to `outputs/refund_ledger.jsonl`.
    5. `escalate_to_human(order_id, reason, summary)`: Routes high-value (> INR 1000) or complex cases to human leads and logs to `outputs/refund_ledger.jsonl`.
- **ReAct Execution Loop (Task 6):**
  - Prompt enforced system trigger `AGENT` and strict compact JSON tool observations (`json.dumps(obs, separators=(',', ':'))`) to prevent token inflation.
  - Evaluated all 7 standard business scenarios from `data/raw/agent_scenarios.csv`:

| Scenario ID | Final Status | Refund Amount (INR) | Model Calls | Tokens Used | Scenario Description & Outcome |
|:---:|:---:|:---:|:---:|:---:|---|
| **S1** | `REFUND` | INR 128.25 | 5 | 789 | Order completed in 52 mins (> 45 min threshold); full refund issued autonomously. |
| **S2** | `REFUND` | INR 135.00 | 5 | 789 | Order completed in 48 mins (> 45 min threshold); full refund issued autonomously. |
| **S3** | `ESCALATE` | INR 1999.00 | 5 | 776 | Order completed in 60 mins; refund exceeds INR 1000 limit and escalated to human lead. |
| **S4** | `DENIED` | INR 0.00 | 4 | 531 | Cancelled order; refund denied per store policy. |
| **S5** | `DENIED` | INR 0.00 | 3 | 296 | Delivered in 15 mins (on-time delivery); refund denied. |
| **S6** | `NOT_FOUND` | INR 0.00 | 2 | 70 | Non-existent order ID in database; agent immediately halted. |
| **S7** | `FALLBACK_MAX_ITERATIONS` | INR 0.00 | 6 | 2655 | Indecisive ambiguous request; agent hit 6-iteration limit and safely halted. |

- **Safe-Stopping Demonstrations (Task 7):**
  1. **Max Iterations Fallback:** Scenario S7 hit the limit of 6 iterations and halted with status `FALLBACK_MAX_ITERATIONS` (6 calls, 2655 tokens).
  2. **Token Limit Fallback:** Scenario S1 executed with `token_limit = 200` halted safely after 3 calls with status `FALLBACK_TOKEN_LIMIT` (313 tokens consumed, strictly guarding against overrun).
  3. **Timeout Fallback:** Scenario S1 executed with `time_limit_seconds = 0` halted immediately before model invocation with status `FALLBACK_TIMEOUT` (0 calls, 0 tokens).
- **Resumable Execution & Crash Recovery (Task 8):**
  - State serialized to `state/<run_id>.json` after every tool execution containing message history, step count, tokens, and executed tools.
  - Simulated a process crash on Scenario S1 immediately after step 2 (tools executed before crash: `['get_order', 'search_policy']`).
  - Executed `resume_agent("s1_resumable_demo")`: loaded persisted state, resumed execution without repeating prior tools, and completed with tools `calculate_refund` and `issue_refund`.
  - Ordered tools executed across the full lifecycle: `get_order` → `search_policy` → `calculate_refund` → `issue_refund` (exactly 4 tools, zero duplicates, final refund: INR 128.25).
- **Idempotency Guarantee:** Both `issue_refund` and `escalate_to_human` verify prior ledger entries before recording, guaranteeing zero duplicate transactions.

---

### Architectural Analysis: ReAct vs. Plan-and-Execute

ReAct (Reason + Act) interleaves thought, action, and observation in a dynamic feedback loop. At each step, the agent inspects intermediate tool outputs before deciding whether to invoke another tool or terminate. In contrast, Plan-and-Execute generates an entire sequence of actions upfront and runs them deterministically before evaluating the final state.

For customer support and refund workflows, ReAct is significantly more adaptable and token-efficient because real-world execution paths depend entirely on dynamic database states. In our evaluation, Scenario S1 required 5 model calls to fetch the order, search policy, calculate the refund, and issue the credit. However, Scenario S5 (an order already delivered in 15 minutes) was identified as non-refundable immediately after checking policy, terminating safely in just 3 calls without wasting LLM invocations on calculation or execution. Scenario S6 (a non-existent order) halted after only 2 calls upon receiving a database not-found observation. A Plan-and-Execute agent would have generated a rigid multi-step plan assuming order existence, requiring complex replanning loops and unnecessary token expenditure when assumptions failed.

Conversely, Plan-and-Execute excels in predictable, multi-goal environments with known dependencies where upfront global optimization minimizes latency. For autonomous refund agents operating under strict financial guardrails and unpredictable human inputs, ReAct provides superior conditional branching, precise tool-level error handling, and minimal token waste.

# Limitations and Next Steps

- **Post-Hoc Premium Label Limitation:** The Premium flag comes from the whole-window RFM of Task 3, which includes the September window that defines the lapse label, so it is a post-hoc whole-window label: the retention list is illustrative and not a leakage-free evaluation.
- **Pre-Cutoff Predictor Rigor:** While the retention list segment label is post-hoc, all input features for the lapse model (`recency`, `frequency`, `monetary`, `avg_discount_pct`, `cancellation_rate`, `avg_delivery_minutes`, `favourite_category`, `city`, `age`, `gender`, `tenure_days`) were engineered strictly using pre-cutoff data without leakage.
- **Future Enhancements:** Future iterations could transition to rolling dynamic window RFM scoring and integrate survival analysis to estimate time-to-lapse alongside binary lapse probabilities.


