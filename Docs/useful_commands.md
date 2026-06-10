Start llama.cpp
``` bash
C:\Code\llamacpp\llama.cpp\build\bin\Release\llama-server.exe -m C:\models\Pixtral-12B-2409-Q4_K_S.gguf --mmproj "C:\models\mmproj-Pixtral-12B-2409-F16.gguf" -ngl 99 --port 8080 --ctx-size 8192 --no-jinja --chat-template chatml --cache-ram 0

C:\Code\llamacpp\llama.cpp\build\bin\Release\llama-server.exe -m "C:\models\qwen3.5-14b-a3b-claude-4.6-opus-reasoning-distilled-reap-q4_k_m.gguf" -ngl 99 --port 8080 --ctx-size 8192 --no-jinja --chat-template chatml --cache-ram 0
```

Venv
``` bash
.\.venv\Scripts\activate
```