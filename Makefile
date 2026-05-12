.PHONY: help track skeleton commit version goals setup clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ── Pipeline ───────────────────────────────────────────────────────────

track: ## Run hand tracking on data/input.mp4
	python src/track_hands.py data/input.mp4

skeleton: ## Run skeleton estimation on data/input.mp4
	python src/estimate_skeleton.py data/input.mp4

# ── Workflow ───────────────────────────────────────────────────────────

commit: ## Auto-commit and push to GitHub
	bash scripts/commit_and_push.sh

version: ## Create a new version (usage: make version V=0.1 T="Title")
	bash scripts/new_version.sh $(V) "$(T)"

goals: ## Show current goals
	@echo ""
	@echo "━━━ Current Goals ━━━"
	@cat goals/current_goals.yaml
	@echo ""

# ── Setup ──────────────────────────────────────────────────────────────

setup: ## Set up the development environment
	bash scripts/setup_env.sh

clean: ## Remove generated outputs (keeps data/)
	rm -rf outputs/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
