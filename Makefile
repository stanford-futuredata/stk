clean:
	rm -rf dist/*

dist: clean
	python3 setup.py sdist

upload: dist
	twine upload dist/*
