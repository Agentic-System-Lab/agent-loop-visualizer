"""What runs the model, behind the one interface the harness and the app know.

backend.py is that interface (Backend), with the settings and the encoded prompt a step passes it. simple_backend.py
runs the model on this machine with transformers and a hand-written decoding loop; vllm_backend.py asks a vLLM server.
"""
