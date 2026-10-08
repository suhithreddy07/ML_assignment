#!/usr/bin/env bash
# Runs the complete polynomial-regression pipeline (about 8-10 minutes on one CPU core).
set -e
pip install -r requirements.txt
python train_models.py     # final models -> prediction CSVs, results.json
python compare_models.py   # OLS / Ridge / Lasso / ElasticNet comparison -> compare_results.jsonl
python shift_check.py      # var1 inner/outer extrapolation check -> shift_results.json
python make_figures.py     # model-comparison, shift and data figures -> figures/
python diagnostics.py      # train-vs-CV and final residual diagnostic figures -> figures/
# optional: compile the LaTeX report (needs a LaTeX installation)
if command -v pdflatex >/dev/null; then (cd report && pdflatex -interaction=nonstopmode report.tex >/dev/null && pdflatex -interaction=nonstopmode report.tex >/dev/null); fi
