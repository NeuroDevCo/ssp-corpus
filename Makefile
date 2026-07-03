sync:
	uv sync

langs: sync
	uv run main.py -l

words: sync
	uv run main.py -w

sylls: sync
	uv run main.py -s

model: sync
	uv run main.py -m

viz: sync
	uv run main.py -v

docs: sync
	uv run main.py -d

all:
	uv run main.py -l -w -s -m -v -d


