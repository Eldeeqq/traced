.PHONY: test
test:
	PYTHONPATH=. pytest

requirements:
	echo "Generating requirements.txt using current env..."
	pip list --format=freeze > requirements.txt
