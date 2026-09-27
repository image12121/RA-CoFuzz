PYTHON ?= python
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: verify-public verify-extensions figures manifest verify-manifest verify-release

verify-public:
	$(PYTHON) -B -m unittest discover -s tests -p 'test_*.py'

verify-extensions:
	$(PYTHON) -B -m unittest discover -s extensions_q20 -p 'test_*.py'

figures:
	$(PYTHON) -B analysis/make_figures.py \
		--summary results/aggregate/summary.json \
		--output-dir figures

manifest:
	find . -type f \
		! -path './MANIFEST.sha256' \
		! -path '*/__pycache__/*' \
		! -path '*/.ipynb_checkpoints/*' \
		! -name '*.py[co]' -print0 \
		| sort -z | xargs -0 sha256sum > MANIFEST.sha256

verify-manifest:
	sha256sum -c MANIFEST.sha256

verify-release: verify-public verify-extensions verify-manifest
	$(PYTHON) -B extensions_q20/run.py audit
	$(PYTHON) -B analysis/make_figures.py \
		--summary results/aggregate/summary.json \
		--output-dir /tmp/ra-cofuzz-release-figures
