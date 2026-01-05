optimize-gru:
	uv run python -m scripts.hp_optimization.optimize_gru
optimize-hgru:
	uv run python -m scripts.hp_optimization.optimize_hgru
optimize-multi-gru:
	uv run python -m scripts.hp_optimization.optimize_multi_gru

run-deploy:
	uv run python -m deployment.deployment_pipeline

sl:
	uv run streamlit run streamlit_app/app.py

