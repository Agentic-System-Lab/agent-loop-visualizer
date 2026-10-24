"""運作說明: the second page of the app, a walk through the agent loop with the code behind it.
The content lives in guide.py."""

import streamlit as st

import guide
import render

diagram = render.diagram_html(None, dark=st.context.theme.type == "dark", counter="每一圈 = 一個 step", model="Gemma 4 · Qwen3.5")
st.html(guide.page_html(diagram))
