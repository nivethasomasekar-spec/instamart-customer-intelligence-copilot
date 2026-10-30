"""
segmentation.py -- Part B Module B1: Customer RFM & Segmentation Pipeline

Tasks implemented:
  - Task 1: Customer RFM Table construction & validation as of 2025-10-01 00:00:00.
  - Task 2: Standardizing RFM features & evaluating K-Means clustering for K=2..8.
  - Task 3: Segment profiling, dynamic business naming, PCA projection & export.
"""

import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def build_rfm_table(orders_csv_path: str = "data/clean/orders.csv") -> pd.DataFrame:
    """
    Builds the customer RFM table as of 2025-10-01 00:00:00 using completed orders only:
      - Recency: whole days between snapshot and latest completed order.
      - Frequency: count of distinct completed orders.
      - Monetary: total spend on completed orders.
    """
    assert os.path.exists(orders_csv_path), f"Missing orders file: {orders_csv_path}"

    # 1. Load clean orders dataset
    df_orders = pd.read_csv(orders_csv_path)
    df_orders["order_ts"] = pd.to_datetime(df_orders["order_ts"])

    # 2. Filter for completed orders only
    completed = df_orders[df_orders["status"] == "completed"].copy()

    # 3. Set snapshot moment
    snapshot = pd.Timestamp("2025-10-01 00:00:00")

    # 4. Group by customer and compute metrics
    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )

    # 5. Compute recency in whole integer days
    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days

    # 6. Keep clean columns
    rfm = rfm[["customer_id", "recency", "frequency", "monetary"]].copy()

    return rfm


def validate_rfm(rfm: pd.DataFrame):
    """
    Validates the RFM table against all 8 official reference checkpoints.
    """
    print("=" * 75)
    print("PART B TASK 1: VALIDATING RFM TABLE AGAINST OFFICIAL CHECKPOINTS")
    print("=" * 75)

    # 1. Customer count
    cust_count = len(rfm)
    print(f"Total customers with completed orders: {cust_count}")
    assert cust_count == 1175, f"Expected 1,175 customers, got {cust_count}"
    print(">> Checkpoint 1/8 [Customer Count]: PASS (1,175 customers)")

    # 2-4. Recency metrics
    rec_median = int(rfm["recency"].median())
    rec_min = int(rfm["recency"].min())
    rec_max = int(rfm["recency"].max())
    print(f"Recency   -> Median: {rec_median} days, Min: {rec_min} days, Max: {rec_max} days")
    assert rec_median == 19, f"Expected recency median 19, got {rec_median}"
    assert rec_min == 0, f"Expected recency min 0, got {rec_min}"
    assert rec_max == 180, f"Expected recency max 180, got {rec_max}"
    print(">> Checkpoint 2/8 [Recency Median]: PASS (19 days)")
    print(">> Checkpoint 3/8 [Recency Min]: PASS (0 days)")
    print(">> Checkpoint 4/8 [Recency Max]: PASS (180 days)")

    # 5-6. Frequency metrics
    freq_mean = round(float(rfm["frequency"].mean()), 2)
    freq_max = int(rfm["frequency"].max())
    print(f"Frequency -> Mean: {freq_mean} orders, Max: {freq_max} orders")
    assert freq_mean == 9.46, f"Expected frequency mean 9.46, got {freq_mean}"
    assert freq_max == 33, f"Expected frequency max 33, got {freq_max}"
    print(">> Checkpoint 5/8 [Frequency Mean]: PASS (9.46 orders)")
    print(">> Checkpoint 6/8 [Frequency Max]: PASS (33 orders)")

    # 7-8. Monetary metrics
    mon_mean = round(float(rfm["monetary"].mean()), 2)
    mon_max = round(float(rfm["monetary"].max()), 2)
    print(f"Monetary  -> Mean: INR {mon_mean:,.2f}, Max: INR {mon_max:,.2f}")
    assert mon_mean == 2552.06, f"Expected monetary mean 2552.06, got {mon_mean}"
    assert mon_max == 14766.55, f"Expected monetary max 14766.55, got {mon_max}"
    print(">> Checkpoint 7/8 [Monetary Mean]: PASS (INR 2,552.06)")
    print(">> Checkpoint 8/8 [Monetary Max]: PASS (INR 14,766.55)")

    print("\n" + "=" * 75)
    print("ALL 8 RFM REFERENCE CHECKPOINTS PASSED PERFECTLY!")
    print("=" * 75)


def run_kmeans_evaluation(rfm: pd.DataFrame):
    """
    Part B Task 2: Evaluates K-Means clustering for K from 2 through 8.
      1. Standardizes RFM columns using StandardScaler.
      2. Fits KMeans(n_clusters=K, n_init=10, random_state=42) for K in [2, 8].
      3. Records inertia and silhouette score.
      4. Validates every K against official reference targets within tolerances.
      5. Confirms K=4 achieves the peak silhouette score.
      6. Returns the fitted scaler, scaled data, K=4 model, and labels for Task 3.
    """
    print("\n" + "=" * 75)
    print("PART B TASK 2: K-MEANS CLUSTERING EVALUATION (K = 2 to 8)")
    print("=" * 75)

    # 1. Extract numerical features and scale
    X_rfm = rfm[["recency", "frequency", "monetary"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_rfm)

    # Official reference values: K -> (expected_inertia, expected_silhouette)
    # Tolerances: +-5% on inertia, +-0.05 on silhouette
    ref_table = {
        2: (1820.8, 0.465),
        3: (1268.2, 0.445),
        4: (743.1, 0.526),
        5: (617.4, 0.481),
        6: (503.2, 0.482),
        7: (408.5, 0.416),
        8: (366.5, 0.412),
    }

    eval_rows = []
    models = {}
    labels_dict = {}

    print(f"\n{'K':<4} | {'Inertia':<10} | {'Ref Inertia':<12} | {'Silhouette':<12} | {'Ref Silhouette':<15} | {'Status'}")
    print("-" * 75)

    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        cluster_labels = km.fit_predict(X_scaled)
        
        inertia = round(float(km.inertia_), 1)
        sil = round(float(silhouette_score(X_scaled, cluster_labels)), 3)
        
        models[k] = km
        labels_dict[k] = cluster_labels
        
        exp_inertia, exp_sil = ref_table[k]
        
        # Verify within stated tolerances
        inertia_pct_diff = abs(inertia - exp_inertia) / exp_inertia
        sil_diff = abs(sil - exp_sil)
        
        assert inertia_pct_diff <= 0.05, f"K={k} inertia {inertia} outside +-5% of {exp_inertia}"
        assert sil_diff <= 0.05, f"K={k} silhouette {sil} outside +-0.05 of {exp_sil}"
        
        eval_rows.append({
            "K": k,
            "Inertia": inertia,
            "Ref_Inertia": exp_inertia,
            "Silhouette": sil,
            "Ref_Silhouette": exp_sil,
        })
        
        status = "PASS (Optimal)" if k == 4 else "PASS"
        print(f"{k:<4} | {inertia:<10} | {exp_inertia:<12} | {sil:<12} | {exp_sil:<15} | {status}")

    print("-" * 75)

    # Confirm K=4 has the highest silhouette score across all tested K
    df_eval = pd.DataFrame(eval_rows)
    best_k = df_eval.loc[df_eval["Silhouette"].idxmax(), "K"]
    print(f"\nCluster count with highest silhouette score: K = {best_k}")
    assert best_k == 4, f"Expected K=4 to have highest silhouette, got K={best_k}"
    print(">> Checkpoint: K=4 confirmed as global silhouette peak (0.526) and elbow point!")

    # Return elements needed for Task 3
    kmeans_4 = models[4]
    labels_4 = labels_dict[4]

    return scaler, X_scaled, kmeans_4, labels_4, df_eval


def profile_and_project_segments(
    rfm: pd.DataFrame, X_scaled, kmeans_4: KMeans, labels_4
) -> pd.DataFrame:
    """
    Part B Task 3: Segment Profiling, Business Naming, PCA & File Export.
      1. Profiles the 4 clusters by size, mean recency, mean frequency, and mean monetary.
      2. Dynamically maps clusters to business names based on their actual RFM metrics:
           - Highest mean monetary -> "High-Value/Premium Buyers" (is_premium = True)
           - Highest mean frequency -> "Frequent Daily Shoppers"
           - Highest mean recency -> "At-Risk/Dormant Shoppers"
           - Remaining cluster -> "Occasional Grocery Buyers"
      3. Applies PCA(n_components=2) and validates explained variance ratios (~0.634, ~0.218, total ~0.852).
      4. Writes outputs/customer_segments.csv with required columns.
    """
    print("\n" + "=" * 75)
    print("PART B TASK 3: SEGMENT PROFILING, BUSINESS PERSONAS & PCA")
    print("=" * 75)

    df_seg = rfm.copy()
    df_seg["segment_id"] = labels_4

    # 1. Profile clusters by their raw RFM statistics
    raw_profile = (
        df_seg.groupby("segment_id")
        .agg(
            size=("customer_id", "count"),
            mean_recency=("recency", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean"),
        )
        .round(2)
    )

    print("\nCluster RFM Profile (Raw K-Means Labels):")
    print(raw_profile)

    # 2. Dynamic Business Mapping based on behavioral metrics (not assumed cluster numbers)
    # (a) Premium: segment with highest mean monetary value
    premium_id = raw_profile["mean_monetary"].idxmax()

    # (b) Frequent Shoppers: highest mean frequency (among non-premium)
    remaining_after_prem = raw_profile.drop(index=premium_id)
    frequent_id = remaining_after_prem["mean_frequency"].idxmax()

    # (c) At-Risk: highest mean recency (among remaining)
    remaining_after_freq = remaining_after_prem.drop(index=frequent_id)
    at_risk_id = remaining_after_freq["mean_recency"].idxmax()

    # (d) Occasional: the remaining cluster
    occasional_id = remaining_after_freq.drop(index=at_risk_id).index[0]

    name_map = {
        premium_id: "High-Value/Premium Buyers",
        frequent_id: "Frequent Daily Shoppers",
        at_risk_id: "At-Risk/Dormant Shoppers",
        occasional_id: "Occasional Grocery Buyers",
    }

    action_map = {
        "High-Value/Premium Buyers": "Assign VIP support desk routing and early access to high-margin electronics.",
        "Frequent Daily Shoppers": "Promote daily essentials subscription streaks and category cross-selling coupons.",
        "Occasional Grocery Buyers": "Send weekend promotional bundles and minimum-order free delivery nudges.",
        "At-Risk/Dormant Shoppers": "Trigger targeted win-back notifications with steep reactivation discounts.",
    }

    df_seg["segment_name"] = df_seg["segment_id"].map(name_map)
    df_seg["is_premium"] = df_seg["segment_id"] == premium_id

    # Print named business segment profile table
    named_profile = (
        df_seg.groupby(["segment_id", "segment_name", "is_premium"])
        .agg(
            size=("customer_id", "count"),
            mean_recency=("recency", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean"),
        )
        .round(2)
        .reset_index()
    )

    print("\nNamed Segment Profiles & Business Summary:")
    for _, row in named_profile.iterrows():
        name = row["segment_name"]
        print(f"\n* Segment {row['segment_id']}: {name} (is_premium={row['is_premium']})")
        print(f"  Size: {row['size']} customers ({row['size'] / len(df_seg) * 100:.1f}%)")
        print(f"  RFM Means -> Recency: {row['mean_recency']} days, Frequency: {row['mean_frequency']} orders, Monetary: INR {row['mean_monetary']:,.2f}")
        print(f"  Recommended Action: {action_map[name]}")

    # Validate segment statistics against reference checkpoints from the brief
    # Reference for K=4: sizes about 436 / 138 / 419 / 182
    # Premium segment (~138 customers): monetary ~7,130, frequency ~6.8
    # Frequent segment (~419 customers): frequency ~19.4, recency ~8.8 days
    prem_stats = named_profile[named_profile["segment_name"] == "High-Value/Premium Buyers"].iloc[0]
    freq_stats = named_profile[named_profile["segment_name"] == "Frequent Daily Shoppers"].iloc[0]

    print("\nValidating Segment Profiling Checkpoints against official brief:")
    assert abs(prem_stats["size"] - 138) <= 15, f"Premium size mismatch: {prem_stats['size']}"
    assert abs(prem_stats["mean_monetary"] - 7130) <= 350, f"Premium monetary mismatch: {prem_stats['mean_monetary']}"
    assert abs(prem_stats["mean_frequency"] - 6.8) <= 1.0, f"Premium frequency mismatch: {prem_stats['mean_frequency']}"
    print(f">> Checkpoint [Premium Segment]: PASS (size={prem_stats['size']}, mean_monetary=INR {prem_stats['mean_monetary']:,.2f}, mean_freq={prem_stats['mean_frequency']})")

    assert abs(freq_stats["size"] - 419) <= 25, f"Frequent size mismatch: {freq_stats['size']}"
    assert abs(freq_stats["mean_frequency"] - 19.4) <= 1.5, f"Frequent frequency mismatch: {freq_stats['mean_frequency']}"
    assert abs(freq_stats["mean_recency"] - 8.8) <= 1.5, f"Frequent recency mismatch: {freq_stats['mean_recency']}"
    print(f">> Checkpoint [Frequent Daily Segment]: PASS (size={freq_stats['size']}, mean_freq={freq_stats['mean_frequency']}, mean_recency={freq_stats['mean_recency']} days)")

    # 3. PCA Projection (2 components)
    print("\n--- Applying PCA(n_components=2) ---")
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_scaled)

    evr = pca.explained_variance_ratio_
    evr_1 = round(float(evr[0]), 3)
    evr_2 = round(float(evr[1]), 3)
    evr_total = round(float(evr.sum()), 3)

    print(f"PCA Component 1 Explained Variance Ratio: {evr_1:.3f} (Reference: ~0.634)")
    print(f"PCA Component 2 Explained Variance Ratio: {evr_2:.3f} (Reference: ~0.218)")
    print(f"Total Explained Variance (PC1 + PC2):     {evr_total:.3f} (Reference: ~0.852)")

    # Validate against reference checkpoints within +-0.02
    assert abs(evr_1 - 0.634) <= 0.02, f"PC1 EVR mismatch: {evr_1} != 0.634 +-0.02"
    assert abs(evr_2 - 0.218) <= 0.02, f"PC2 EVR mismatch: {evr_2} != 0.218 +-0.02"
    assert abs(evr_total - 0.852) <= 0.02, f"Total EVR mismatch: {evr_total} != 0.852 +-0.02"
    print(">> Checkpoint [PCA Explained Variance Ratios]: PASS (PC1 ~0.634, PC2 ~0.218, Total ~0.852 within +-0.02)")

    # 4. Save outputs/customer_segments.csv
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "customer_segments.csv")

    cols_order = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]
    df_out = df_seg[cols_order].copy()
    df_out.to_csv(out_csv, index=False)

    print(f"\nSaved segment file to: {out_csv}")
    print(f"File Shape: {df_out.shape} (Expected: 1175 rows, 7 columns)")
    assert df_out.shape == (1175, 7), f"Output shape mismatch: {df_out.shape} != (1175, 7)"
    print(">> Checkpoint [outputs/customer_segments.csv]: PASS (1,175 rows, 7 columns)")

    print("\n" + "=" * 75)
    print("ALL PART B TASK 3 CHECKPOINTS PASSED PERFECTLY!")
    print("=" * 75)

    return df_out, pca


def main():
    # Task 1: RFM Construction & Validation
    rfm = build_rfm_table()
    validate_rfm(rfm)

    # Task 2: K-Means Clustering Evaluation (K = 2 to 8)
    scaler, X_scaled, kmeans_4, labels_4, df_eval = run_kmeans_evaluation(rfm)

    # Task 3: Segment Profiling, Business Naming, PCA & File Export
    df_segments, pca = profile_and_project_segments(rfm, X_scaled, kmeans_4, labels_4)


if __name__ == "__main__":
    main()
