"""The harness: everything that runs a turn, and nothing a person sees.

agent.py is the loop itself (Run): context, model, parse, tool, observation, stop, with one event per stage for the
app to draw. tools.py has the tools it runs, with run_python's Docker sandbox built from sandbox/. media.py turns the
files attached to a question into what the model reads. families/ is everything model-specific, and backends/ is what
runs the model, behind one interface. i18n.py has the page's two languages, which the errors the model reads use too.

Nothing here imports Streamlit, and importing the package loads nothing heavy: the app's guide imports families and
i18n with pygments alone.
"""
