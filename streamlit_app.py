"""Agent Loop 可視化 (Agent Loop Visualizer): where Streamlit starts the app.

Run with:  uv run streamlit run streamlit_app.py

The project has two parts. harness/ runs a turn: the agent loop, the tools it runs, the model's family and what runs
the model; it never imports Streamlit. app/ is what a person sees: the page, the HTML it draws from the harness's
events, and the How it works guide. Streamlit runs this file again on every interaction, and main() draws the page anew.
"""

from app.page import main

main()
