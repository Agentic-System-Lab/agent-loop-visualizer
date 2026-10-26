"""What runs the agent loop, behind the one interface the app knows.

harness.py is that interface (Harness) and the events every harness sends. simple_harness.py is this project's own
harness (SimpleHarness), which runs the model through a Backend (model_backends/); pi_harness.py hands the loop to pi, a
coding agent that runs its own (PiHarness). pi_extension.ts is the pi extension through which the page sees pi's
requests and asks before pi runs a tool, and model_endpoint.py serves the page's own model to pi over HTTP.
"""
