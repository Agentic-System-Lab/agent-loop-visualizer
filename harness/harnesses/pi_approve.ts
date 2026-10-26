/**
 * Ask me before running a tool, for pi run by the Agent Loop Visualizer (harness/harnesses/pi_harness.py).
 *
 * Before pi runs a tool call, this asks through a confirm dialog. In RPC mode the dialog is an extension_ui_request on
 * stdout, which pi_harness.py turns into the page's Allow and Reject buttons; the answer comes back on stdin as an
 * extension_ui_response. A call the user rejects does not run, and the model reads the refusal as its result, in the
 * language the turn started in (AGENT_LOOP_REFUSAL, which pi_harness.py sets).
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const TITLE = "agent-loop:approve"; // pi_harness.APPROVE_TITLE, how pi_harness.py tells this dialog from any other

export default function (pi: ExtensionAPI) {
	pi.on("tool_call", async (event, ctx) => {
		const call = JSON.stringify({ toolCallId: event.toolCallId, toolName: event.toolName, input: event.input });
		if (!(await ctx.ui.confirm(TITLE, call))) {
			return { block: true, reason: process.env.AGENT_LOOP_REFUSAL || "The user refused to run this tool call" };
		}
		return undefined;
	});
}
