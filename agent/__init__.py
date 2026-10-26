"""The agent: everything that runs a turn, and nothing a person sees.

harnesses/ is what runs the loop, behind one interface: simple_harness.py is this project's own loop (SimpleHarness):
context, model, parse, tool, observation, stop, with one event per stage for the app to draw; pi_harness.py hands the
loop to pi, a coding agent, and turns what pi reports into the same events. tools.py has the tools SimpleHarness runs,
with run_python's Docker sandbox built from sandbox/. media.py turns the files attached to a question into what the
model reads. model_families/ is everything model-specific, and model_backends/ is what runs the model, behind one interface.
i18n.py has the page's two languages, which the errors the model reads use too.

Nothing here imports Streamlit, and importing the package loads nothing heavy: the app's guide imports model_families
and i18n with pygments alone.
"""
