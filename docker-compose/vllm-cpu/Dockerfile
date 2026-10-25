# vLLM 0.32.0 on the CPU, which compose.yaml runs when the sidebar picks vLLM (Docker) on a machine without an
# NVIDIA GPU, a Mac say. It is vLLM's own CPU image with two fixes (patch_vllm.py): one from vLLM's main branch that
# came a day too late for 0.32.0, without which no model answers, and one for Gemma 4's audio encoder. The page builds
# it the first time it starts the server; by hand:
#
#     docker compose -f docker-compose/compose.yaml build
#
# compose.yaml tags the image with vLLM's version and this folder's (agent-loop-vllm-cpu:0.32.0-1). Bump the
# second number there when this folder changes, so that a machine with the old image builds the new one.
FROM vllm/vllm-openai-cpu:v0.32.0

COPY patch_vllm.py /tmp/
RUN python3 /tmp/patch_vllm.py && rm /tmp/patch_vllm.py
