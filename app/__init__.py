"""The app: what a person sees, on top of the harness.

page.py is the Streamlit page: the chat on the left, the loop on the right, the sidebar's settings, and the background
thread a turn runs in. render.py turns the harness's events into the HTML the page draws, as pure functions, so the
page can redraw any moment of a run. guide.py builds 運作說明 (How it works), a static page on the loop and the code
behind it. assets/ holds the stylesheets, the logo and the example files.

Nothing here imports torch or transformers at import time but page.py, through the harness's backends: guide.py and
render.py build with pygments alone (.github/workflows/pages.yml).
"""
