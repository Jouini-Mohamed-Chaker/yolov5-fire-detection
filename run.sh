#!/bin/bash

source venv/bin/activate
# pip install -r requirements.txt
uvicorn src.app:app --host 0.0.0.0 --port 8026 --reload