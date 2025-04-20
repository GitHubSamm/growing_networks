# -----------------------------------------------------------------------------
# Makefile
#
# Centralizes and simplifies the reproducibility of the experiments.
#
# Available Commands:
#   make run-basic-MNIST    - Run the basic MNIST training script.
#
# Requirements:
#   - Python environment properly set up (e.g., net2net conda env)
#   - Project installed as editable with `pip install -e .`
#
# -----------------------------------------------------------------------------

PYTHON=python
DEVICE=mps

# Commands

### Basic MNIST ###
run-basic-MNIST:
	$(PYTHON) -m scripts.net2net.basic_MNIST.train


### Net2Net ###

# MNIST
run-net2net-MNIST-with-growth:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model young --dataset MNIST

run-net2net-MNIST-without-growth_young:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model young --dataset MNIST --no_growth_for_baseline

run-net2net-MNIST-without-growth_adult:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model adult --dataset MNIST --no_growth_for_baseline

# Fashion MNIST
run-net2net-FashionMNIST-with-growth:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model young --dataset FASHION_MNIST

run-net2net-FashionMNIST-without-growth_young:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model young --dataset FASHION_MNIST --no_growth_for_baseline

run-net2net-FashionMNIST-without-growth_adult:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model adult --dataset FASHION_MNIST --no_growth_for_baseline



### Sb Recipe ASR TIMIT (MAC OS) ###
run-recipe-asr-timit-mac:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) -m scripts.speech.TIMIT.TIMIT_ASR_sb_recipe.train \
	scripts/speech/TIMIT/TIMIT_ASR_sb_recipe/hparams/train.yaml \
	--data_folder data/raw/TIMIT \
	--device=mps \
	--num_workers=0

## Normal Cuda version ##
run-recipe-asr-timit:
	$(PYTHON) -m scripts.speech.TIMIT.TIMIT_ASR_sb_recipe.train \
	scripts/speech/TIMIT/TIMIT_ASR_sb_recipe/hparams/train.yaml \
	--data_folder data/raw/TIMIT \
	--jit
