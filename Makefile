.PHONY: test
test:
	PYTHONPATH=. pytest

requirements:
	echo "Generating requirements.txt using current env..."
	pip list --not-required --format=freeze > requirements.txt

cleandoc:
	echo "Building docs..."
	cd docs && make html
	echo "Docs built. Open docs/_build/html/index.html in your browser to view."

makedoc:
	echo "Cleaning docs..."
	cd docs && make clean