/**
 * The Agent Loop Visualizer's extension for pi (harness/harnesses/pi_harness.py), loaded into every pi the page starts.
 * It does three things, each set by an environment variable that pi_harness.py sets:
 *
 * - Before every request pi sends a model, it tells the page what the request holds, as a notification, which in RPC
 *   mode is an extension_ui_request on stdout (REQUEST_MARK, then the payload as JSON); the page draws it as the step's
 *   request.
 * - With AGENT_LOOP_ENDPOINT, it registers the page's own model, served by model_endpoint.py, as the provider
 *   agent-loop, so that pi can ask Gemma or Qwen on transformers or vLLM; the thinking level becomes the chat template's
 *   enable_thinking.
 * - With AGENT_LOOP_APPROVE, it asks before every tool call through a confirm dialog, which the page shows as Allow and
 *   Reject; a rejected call does not run, and the model reads the refusal (AGENT_LOOP_REFUSAL) in the turn's language.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const APPROVE_TITLE = "agent-loop:approve"; // pi_harness.APPROVE_TITLE, how pi_harness.py tells this dialog from any other
const REQUEST_MARK = "agent-loop:request "; // pi_harness.REQUEST_MARK
const env = process.env;

export default function (pi: ExtensionAPI) {
	if (env.AGENT_LOOP_ENDPOINT) {
		pi.registerProvider("agent-loop", {
			name: "Agent Loop Visualizer",
			baseUrl: env.AGENT_LOOP_ENDPOINT,
			apiKey: "$AGENT_LOOP_TOKEN",
			api: "openai-completions",
			models: [
				{
					id: env.AGENT_LOOP_MODEL,
					name: env.AGENT_LOOP_MODEL,
					reasoning: true,
					input: (env.AGENT_LOOP_INPUT || "text").split(","),
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					contextWindow: Number(env.AGENT_LOOP_CONTEXT),
					maxTokens: Number(env.AGENT_LOOP_MAX_TOKENS),
					compat: {
						thinkingFormat: "chat-template",
						chatTemplateKwargs: { enable_thinking: { $var: "thinking.enabled" } },
						supportsDeveloperRole: false,
						supportsStore: false,
						supportsReasoningEffort: false,
						maxTokensField: "max_tokens",
					},
				},
			],
		});
	}

	pi.on("before_provider_request", (event, ctx) => {
		ctx.ui.notify(REQUEST_MARK + JSON.stringify(event.payload), "info");
	});

	if (env.AGENT_LOOP_APPROVE) {
		pi.on("tool_call", async (event, ctx) => {
			const call = JSON.stringify({ toolCallId: event.toolCallId, toolName: event.toolName, input: event.input });
			if (!(await ctx.ui.confirm(APPROVE_TITLE, call))) {
				return { block: true, reason: env.AGENT_LOOP_REFUSAL || "The user refused to run this tool call" };
			}
			return undefined;
		});
	}
}
