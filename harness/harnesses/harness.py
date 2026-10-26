"""The interface between the page and whatever runs the agent loop.

The page (app/page.py) runs one turn at a time and reaches the loop only through a Harness: it makes one with a
function that appends each event to the turn's list, calls run() in a background thread, and passes the user's
clicks on with stop() and decide(). render.py draws the list. This project's own harness is
simple_harness.SimpleHarness, which runs the model through a Backend (backends/backend.py), so it knows every token;
pi_harness.PiHarness hands the loop to pi, a coding agent that runs its own and says what it does as JSON. Another
agent that reports its loop as events, such as the Claude Agent SDK or Codex, would be one more; this module says
what it has to send.

A harness that runs the model through a Backend knows the prompt the model reads and each token it writes; one that
asks a model's API sees only the text that comes back. So, like a token event's details (backend.TOKEN_DETAILS), most
events are sent only by a harness that knows what they say, and render.py draws a step from whichever it has: the page
never shows a prompt or a number the harness did not see.

Nothing here imports anything, so a harness can be written, and the interface read, without the rest.
"""

from typing import Protocol, runtime_checkable

# The events every harness sends: run_start first, a step_start each time around the loop, and run_end last (after
# an error event when something failed). run_start carries which harness ran the turn (`harness`: "simple", "pi"), the
# model, its family (a key of families.FAMILIES, or None when the harness does not know how the model writes its turns),
# the turn's settings and the tools the model gets; run_end the stop reason, the final reply and the conversation.
ALWAYS = ("run_start", "step_start", "run_end")
# The events a harness sends in a step when it knows what they say, in the order they come:
#   context            the prompt as the model reads it, from a harness that renders it (simple_harness.SimpleHarness)
#   prefill, token     the prompt read, then each token (backend.TOKEN_FIELDS), from a harness that runs the model itself
#   delta              a piece of the model's output as it streams, with its channel, from one that sees text, not tokens (pi)
#   model_end          the model's turn ended: why, and how many tokens, with their speed when it was measured
#   parsed             what the harness read in it: the thinking, the reply, the tool calls
#   tool_call          a call it will run (or reports as failed), with its environment
#   approval_request, approval   the call held for the user, and the user's answer
#   tool_start, tool_result      the call running in its environment, and what came back
#   error              what went wrong, before run_end
SOMETIMES = ("context", "prefill", "token", "delta", "model_end", "parsed", "tool_call", "approval_request", "approval",
             "tool_start", "tool_result", "error")


@runtime_checkable
class Harness(Protocol):
    """What runs one turn. The page makes one per question (app.page.LiveRun) and runs one at a time."""

    id: str

    def run(self) -> None:
        """Run the turn to its end, sending its events. It returns after run_end and never raises: a failure is an
        error event followed by run_end."""

    def stop(self) -> None:
        """Stop the turn as soon as it can: the user pressed Stop. It is called from another thread, and run() still
        ends with run_end."""

    def decide(self, call_id: str, approved: bool) -> None:
        """The user's answer to the approval_request for the call `call_id`. It is called from another thread."""
