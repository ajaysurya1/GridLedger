"""Plain-language overview of how GridLedger reaches a recommendation."""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("How the review works")
    st.caption("A measured energy gap is a reason to investigate, not proof of wrongdoing.")

    st.graphviz_chart(r"""
    digraph G {
      graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.28", ranksep="0.35"];
      node [shape=box, style="rounded,filled", color="#334155", fillcolor="#111827", fontcolor="#F8FAFC", fontname="sans-serif", fontsize=11, margin="0.14,0.10"];
      edge [color="#60A5FA", penwidth=1.4, arrowsize=0.7];
      collect [label="Collect & clean"];
      balance [label="Energy balance"];
      voltage [label="Voltage check"];
      pattern [label="Usage patterns"];
      reconcile [label="Compare likely causes"];
      fuse [label="Combine evidence"];
      triage [label="Review priority"];
      dispatch [label="Inspection plan"];
      verify [label="Verify outcome"];
      collect -> balance -> voltage -> pattern -> reconcile -> fuse -> triage -> dispatch -> verify;
    }
    """)

    st.subheader("Two calculations at the center")
    st.latex(r"R_t = E_{in,t} - \sum_i E_{meter,i,t} - \widehat{L}_t")
    st.caption("The remaining gap after expected technical losses are subtracted.")
    st.latex(r"\theta^* = \arg\min_{0 \leq \theta_i \leq 1} \|g - A\theta\|_2^2")
    st.caption("The reconciliation step estimates how much of the gap is explained by each customer's measured shortfall.")

    st.subheader("What informed the design")
    st.dataframe([
        {"Research": "Tang, Ten & Schneider (2019)", "What we took / what we changed": "Compare feeder input with downstream meters; adapted as disjoint transformer and feeder balances."},
        {"Research": "Saleem & Weng (2022)", "What we took / what we changed": "Treat meter-to-transformer records as evidence; added a mapping what-if check before recommending field action."},
        {"Research": "Bludszuweit et al. (2022), INTERPRETER", "What we took / what we changed": "Use voltage deviation as a cross-check; model it as an optional corroborating signal, not standalone proof."},
        {"Research": "Orvati Nia et al. (2026)", "What we took / what we changed": "Use temporal baselines; kept interpretable rules in the core and leave GAN-LSTM as an optional future upgrade."},
    ], width="stretch", hide_index=True)

    st.subheader("Assumptions and limits")
    st.warning("""
    - The showcase and benchmark use simulated data, not field results.
    - The voltage check uses a single-phase resistive model.
    - Inspection recommendations support human review; they are not accusations.
    - An unmetered tap and a manipulated meter can look identical in an energy balance.
    - Recovering billed or stolen energy does not by itself save electricity.
    """)