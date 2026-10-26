"""What runs the agent loop, behind the one interface the app knows.

harness.py is that interface (Harness) and the events every harness sends. simple_harness.py is this project's own
harness (SimpleHarness), which runs the model through a Backend (backends/); pi_harness.py hands the loop to pi, a
coding agent that runs its own (PiHarness), and pi_approve.ts is the pi extension through which the page asks before
pi runs a tool.
"""
