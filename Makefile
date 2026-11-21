.PHONY: install corpus run test eval ablation sweep docker
install:  ; pip install -r requirements-dev.txt
corpus:   ; python scripts/build_corpus.py
run:      ; uvicorn app.main:app --reload --port 8000
test:     ; pytest -q
eval:     ; python -m evals.run_eval --label local --gate
ablation: ; python -m evals.run_eval --ablation
sweep:    ; python -m evals.run_eval --sweep
docker:   ; docker compose up --build
