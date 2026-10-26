"""What runs the model, behind the one interface the harness and the app know.

backend.py is that interface (Backend), with the settings and the encoded prompt a step passes it. simple_backend.py
runs the model on this machine with transformers and a hand-written decoding loop; vllm_backend.py asks a vLLM server.
vllm_docker.py starts such a server in Docker when the sidebar picks vLLM without VLLM_URL, from the project's
docker-compose/: compose.yaml (vLLM on the CPU, built from vllm-cpu/) and compose.gpu.yaml (on NVIDIA GPUs).
"""
