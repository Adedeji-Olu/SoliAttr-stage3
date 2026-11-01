# make reproduce DATA=/path/to/SoliData.zip   -> builds cache, runs all baselines (4 folds x 5 repeats), writes results/table2.*
# make smoke                                  -> 2-minute end-to-end check on synthetic data (no download needed)
DATA ?= SoliData.zip
DEVICE ?= cpu
export PYTHONPATH := src

reproduce:
	python scripts/prepare_data.py --src "$(DATA)" --out cache
	python scripts/run_baselines.py --cache cache --out results --device $(DEVICE) --random-split-reference
	python scripts/report.py --res results

stability:
	python scripts/run_stability.py --cache cache --out results --device $(DEVICE)
	python scripts/report.py --res results

test:
	python -m pytest -q tests

smoke:
	python scripts/make_synthetic.py --out .smoke/raw
	python scripts/prepare_data.py --src .smoke/raw --out .smoke/cache > /dev/null
	python scripts/run_baselines.py --cache .smoke/cache --out .smoke/res --repeats 1 --folds 1 --epochs 5 --methods majority bayes svm cnn
	python scripts/run_stability.py --cache .smoke/cache --out .smoke/res --repeats 1 --folds 1 --epochs 30 --max-cohort 30
	python scripts/report.py --res .smoke/res

rq3:
	python scripts/rq3_correlation.py --results results
