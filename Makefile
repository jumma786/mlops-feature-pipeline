.PHONY: install test pipeline mlflow-ui clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

pipeline:
	python src/features/pipeline_builder.py
	python src/train.py

mlflow-ui:
	mlflow ui --backend-store-uri mlruns --port 5000

clean:
	rm -rf mlruns/ artifacts/ __pycache__
	find . -name "*.pyc" -delete

help:
	@echo "Commands:"
	@echo "  make install    Install dependencies"
	@echo "  make test       Run 17 unit tests"
	@echo "  make pipeline   Fit feature pipeline + train model"
	@echo "  make mlflow-ui  Launch MLflow UI"
	@echo "  make clean      Remove generated files"
