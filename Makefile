.PHONY: test
test:
	PYTHONPATH=. pytest

requirements:
	echo "Generating requirements.txt using current env..."
	pip list --not-required --format=freeze > requirements.txt
