ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server
FROM ${LLAMA_IMAGE}

COPY models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf /models/Qwen3-0.6B-Q8_0.gguf

EXPOSE 8001

ENTRYPOINT ["llama-server"]
CMD [
  "--model", "/models/Qwen3-0.6B-Q8_0.gguf",
  "--alias", "Qwen/Qwen3-0.6B",
  "--host", "0.0.0.0",
  "--port", "8001",
  "--ctx-size", "8192",
  "--parallel", "1",
  "--gpu-layers", "0",
  "--reasoning", "off",
  "--jinja"
]
