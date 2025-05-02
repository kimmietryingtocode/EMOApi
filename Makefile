run:
	uvicorn app.main:app --reload

lint:
	ruff check app

format:
	ruff format app

test:
	pytest

freeze:
	pip freeze > requirements.txt
