.PHONY: install test train api web docker clean

install:
	pip install -r requirements.txt

test:
	cd app && python3 test.py

baseline:
	cd app && python -c "from train import train_baseline; train_baseline()"

hybrid:
	cd app && python -c "from train import train_hybrid; train_hybrid()"

train:
	cd app && python train.py

api:
	cd app && uvicorn api:app --reload --port 8000

web:
	cd app && python web.py

docker:
	docker-compose up

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -exec rm -rf {} +