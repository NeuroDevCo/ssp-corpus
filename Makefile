main:
	python -m main

venv:
	python -m venv .env

freeze:
	python -m pip freeze -l > requirements.txt 

install:
	python -m pip install -r requirements.txt


documentation:
	quarto render docs/index.qmd --to pdf
	quarto render docs/index.qmd --to docx
