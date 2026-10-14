#!/bin/bash
# Started by the Lambda Web Adapter (AWS_LAMBDA_EXEC_WRAPPER=/opt/bootstrap).
export PYTHONPATH="/var/task/app:$PYTHONPATH"
# libllama and the ggml libraries ship in the zip next to the shim.
export LD_LIBRARY_PATH="/var/task/app/libs:$LD_LIBRARY_PATH"
exec python -m uvicorn --app-dir /var/task/app main:app --host 127.0.0.1 --port "${PORT:-8080}" --workers 1
