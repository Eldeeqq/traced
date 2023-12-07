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

lint:
	echo "Linting..."
	flake8 traced_v2 --ignore=E501,E722,E731,W503,W504
	black traced_v2
	ruff traced_v2
	isort traced_v2
	echo "Linting done."