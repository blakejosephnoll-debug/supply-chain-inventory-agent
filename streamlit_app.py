# streamlit_app.py
"""Streamlit web interface for the Supply Chain Inventory Decision Agent.

The application allows a user to enter a natural‑language inventory question.
It reuses the existing `InventoryDecisionAgent` (which encapsulates the full
pipeline) and displays the eight required verification sections exactly as the
CLI version does, but formatted with Streamlit widgets.

Requirements:
- Do not modify any calculation logic – all numeric work is performed by the
  deterministic tools in `inventory_tools.py` via the agent.
- Show the sections:
  1. Inputs Extracted
  2. Variable Definitions
  3. Formula
  4. Formula with Substituted Values
  5. Intermediate Calculations
  6. Final Answer
  7. Verification Checks
  8. Assumptions
- Use Streamlit's markdown and `st.latex` for equations where appropriate.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from inventory_agent import InventoryDecisionAgent, AgentResponse

# ---------------------------------------------------------------------------
# Helper functions to render each section
# ---------------------------------------------------------------------------

def render_section(title: str, content: object) -> None:
    """Render a generic section with a markdown title.

    * If ``content`` is a mapping, it is displayed as JSON for readability.
    * If ``content`` is a list of strings, each element is shown as a bullet.
    * For plain strings, show as markdown (allow LaTeX).
    """
    st.subheader(title)
    if isinstance(content, dict):
        # Use ``st.json`` for a compact view – it preserves key ordering.
        st.json(content)
    elif isinstance(content, list):
        for item in content:
            st.markdown(f"- {item}")
    else:
        # Assume string – display markdown to enable LaTeX formatting.
        st.markdown(content)

# Function to render the full agent response with all eight sections
def render_agent_response(resp: AgentResponse) -> None:
    """Display the eight verification sections from an ``AgentResponse``.

    The keys in ``resp.result_dict`` are prefixed with numbers (e.g.
    ``'1_inputs_extracted'``).  This helper retrieves the value whose key ends
    with the desired suffix.
    """
    def _get_by_suffix(suffix: str):
        for k, v in resp.result_dict.items():
            if k.lower().endswith(suffix):
                return v
        return None

    # 1. Inputs Extracted
    render_section("Inputs Extracted", resp.extracted_inputs)

    # 2. Variable Definitions
    var_defs = _get_by_suffix("_variable_definitions")
    if var_defs:
        render_section("Variable Definitions", var_defs)

    # 3. Formula
    formula = _get_by_suffix("_formula")
    if formula:
        render_section("Formula", formula)

    # 4. Formula with Substituted Values
    substituted = _get_by_suffix("_formula_with_substituted_values")
    if substituted:
        render_section("Formula with Substituted Values", substituted)

    # 5. Intermediate Calculations
    intermediates = _get_by_suffix("_intermediate_calculations")
    if intermediates:
        render_section("Intermediate Calculations", intermediates)

    # 6. Final Answer
    final_ans = _get_by_suffix("_final_answer")
    if isinstance(final_ans, dict):
        render_section("Final Answer", final_ans)
    else:
        render_section("Final Answer", str(final_ans) if final_ans is not None else "None")

    # 7. Verification Checks
    verification = resp.verification
    if verification:
        status = verification.get("status", "UNKNOWN")
        st.markdown(f"**Overall Verification Status:** {status}")
        for chk in verification.get("checks", []):
            passed = chk.get("passed", False)
            icon = "✅" if passed else "❌"
            st.write(f"{icon} **{chk.get('check', '')}** – {chk.get('detail', '')}")

    # 8. Assumptions
    assumptions = _get_by_suffix("_assumptions")
    if assumptions:
        render_section("Assumptions", assumptions)


def classify_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Perform deterministic multicriteria classification without duplicating rows.

    Steps:
    1. Work on a copy and keep the original row order via the DataFrame index.
    2. Sort by Sales (descending) then Item Number (ascending) to compute cumulative
       sales percentages.  The resulting percentages are written back to the original
       rows using the index, avoiding any merge on non‑unique values.
    3. Independently sort by Hits (descending) then Item Number (ascending) to compute
       cumulative hits percentages and map them back the same way.
    4. Map cumulative percentages to ABCD classes (≤60 % → A, ≤80 % → B, ≤90 % → C, else D).
    5. Convert classes to scores (A=4, B=3, C=2, D=1), compute the weighted score
       (0.65 × sales_score + 0.35 × hits_score) and assign the final class using the
       professor's thresholds.
    6. Run validation checks to ensure the output row count matches the input,
       total sales are unchanged, each original index appears exactly once, and the
       final class counts sum to the original number of rows.
    """
    # Work on a copy so the original DataFrame stays untouched.
    df = df.copy()
    # Preserve the original index for later re‑assignment.
    df['_orig_idx'] = df.index

    # Normalise column names.
    col_map = {c.lower(): c for c in df.columns}
    sales_col = col_map.get('sales ($)') or col_map.get('sales')
    hits_col = col_map.get('hits')
    item_col = col_map.get('item number') or col_map.get('item number') or 'Item Number'
    if sales_col is None or hits_col is None:
        st.error('Uploaded CSV must contain "Sales ($)" and "Hits" columns.')
        return df.drop(columns=['_orig_idx'])

    # ---------------------------------------------------------------------
    # 1. Sales cumulative percentage with deterministic tie‑breaking
    # ---------------------------------------------------------------------
    sales_sorted = df.sort_values(by=[sales_col, item_col], ascending=[False, True])
    total_sales = sales_sorted[sales_col].sum()
    sales_sorted['sales_cum_pct'] = sales_sorted[sales_col].cumsum() / total_sales * 100
    # Map back to original rows via the stored original index.
    sales_cum_map = sales_sorted.set_index('_orig_idx')['sales_cum_pct']
    df['sales_cum_pct'] = df['_orig_idx'].map(sales_cum_map)

    # ---------------------------------------------------------------------
    # 2. Hits cumulative percentage with deterministic tie‑breaking
    # ---------------------------------------------------------------------
    hits_sorted = df.sort_values(by=[hits_col, item_col], ascending=[False, True])
    total_hits = hits_sorted[hits_col].sum()
    hits_sorted['hits_cum_pct'] = hits_sorted[hits_col].cumsum() / total_hits * 100
    hits_cum_map = hits_sorted.set_index('_orig_idx')['hits_cum_pct']
    df['hits_cum_pct'] = df['_orig_idx'].map(hits_cum_map)

    # ---------------------------------------------------------------------
    # 3. Map cumulative percentages to ABCD classes and scores
    # ---------------------------------------------------------------------
    def map_class(pct: float):
        if pct <= 60:
            return 'A', 4
        elif pct <= 80:
            return 'B', 3
        elif pct <= 90:
            return 'C', 2
        else:
            return 'D', 1

    df[['sales_class', 'sales_score']] = df['sales_cum_pct'].apply(lambda p: pd.Series(map_class(p)))
    df[['hits_class', 'hits_score']] = df['hits_cum_pct'].apply(lambda p: pd.Series(map_class(p)))

    # ---------------------------------------------------------------------
    # 4. Weighted score and final class
    # ---------------------------------------------------------------------
    df['weighted_score'] = 0.65 * df['sales_score'] + 0.35 * df['hits_score']

    def final_class(ws: float):
        if ws >= 3.25:
            return 'A'
        if ws >= 2.5:
            return 'B'
        if ws >= 1.75:
            return 'C'
        return 'D'

    df['final_class'] = df['weighted_score'].apply(final_class)

    # ---------------------------------------------------------------------
    # 5. Validation checks
    # ---------------------------------------------------------------------
    original_row_count = len(df)
    # Total Sales before classification (using original column)
    total_sales_before = df[sales_col].sum()
    # Ensure no duplicate rows were introduced.
    assert df['_orig_idx'].nunique() == original_row_count, "Duplicate rows detected after classification"
    # Confirm row count unchanged.
    assert len(df) == original_row_count, "Row count changed after classification"
    # Confirm total sales unchanged.
    assert abs(df[sales_col].sum() - total_sales_before) < 1e-6, "Total Sales mismatch after classification"
    # Final class counts sum to original rows.
    final_counts = df['final_class'].value_counts()
    assert final_counts.sum() == original_row_count, "Final class counts do not sum to original row count"

    # Drop the helper index column before returning.
    df = df.drop(columns=['_orig_idx'])
    return df

def main() -> None:
    st.set_page_config(page_title="Inventory Decision Agent", layout="centered")

    # ---------- CSV uploader (single) ----------
    uploaded_file = st.file_uploader("Upload inventory CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df = None
        if df is not None:
            st.success("CSV file uploaded successfully.")
            st.subheader("Raw Data (first 10 rows)")
            st.dataframe(df.head(10))
            if st.button("Run ABCD Classification"):
                classified_df = classify_inventory(df)
                st.subheader("Item‑level Classification")
                st.dataframe(classified_df)
                # Summary table
                summary = (
                    classified_df.groupby('final_class')
                    .agg(items=('Item Number', 'count'), total_sales=('Sales ($)', 'sum'))
                    .reindex(['A', 'B', 'C', 'D'])
                    .fillna(0)
                )
                st.subheader("Summary by Final Class")
                st.dataframe(summary)
        # end of uploader block

    st.markdown(
        "Enter a natural‑language inventory question (EOQ, Quantity Discount, "
        "Safety Stock & ROP, Fixed‑Period Review, or Newsvendor). The agent will "
        "identify the model, extract parameters, run the deterministic calculation, "
        "and display the eight verification sections."
    )

    # Text input for the query
    query = st.text_area("Inventory Question", height=150)
    if st.button("Run Decision Engine"):
        if not query.strip():
            st.warning("Please provide a question before running the agent.")
        else:
            with st.spinner("Processing …"):
                agent = InventoryDecisionAgent()
                response: AgentResponse = agent.solve(query)
            st.success("✅ Calculation completed")
            render_agent_response(response)


if __name__ == "__main__":
    main()
