#!/bin/bash

# Démarrer l'API FastAPI avec uvicorn 
uvicorn main:app --host=0.0.0.0 --port=8000
