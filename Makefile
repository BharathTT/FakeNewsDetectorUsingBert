.PHONY: install test train api web docker clean

install:
	pip install -r requirements.txt

test:
	cd app && python3 test.py

baseline:
	cd app && python3 -c "from train import train_baseline; train_baseline()"

train:
	cd app && python3 train.py

api:
	cd app && uvicorn api:app --reload --port 8000

web:
	cd app && python3 web.py

docker:
	docker-compose up

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -exec rm -rf {} +