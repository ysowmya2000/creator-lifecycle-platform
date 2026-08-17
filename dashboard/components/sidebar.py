"""Shared sidebar: project title, description, and links."""

import streamlit as st


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🎙 Creator Lifecycle Intelligence")
        st.caption(
            "Survival analysis on 168K podcast creators — cold start prediction, "
            "monetization causal inference, and competing-risks dormancy classification."
        )
        st.markdown("---")
        st.markdown("**Modules**")
        st.markdown(
            "1. Survival Explorer\n"
            "2. Early Warning Simulator\n"
            "3. Monetization Impact\n"
            "4. Dormancy Intelligence"
        )
        st.markdown("---")
        st.markdown("[GitHub repo](https://github.com/ysowmya2000/creator-lifecycle-platform)")
        st.caption(
            "Data: [SPoRC](https://huggingface.co/datasets/blitt/SPoRC) "
            "(research-use-only license). See data/README.md for the windowing "
            "and reframing notes that shaped every metric here."
        )
