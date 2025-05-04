#!/bin/bash
uvicorn main:app --reload &
streamlit run app.py
