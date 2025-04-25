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
###################
### Basic MNIST ###
###################
run-basic-MNIST:
	$(PYTHON) -m scripts.net2net.basic_MNIST.train

###############
### Net2Net ###
###############
# Change strat to test variations:
#---- Strat 1 = Wider
#---- Strat 2 = Deeper
#---- Strat 3 = Wider + BatchNorm handling


#==== MNIST ====#
# Strat 1
run-net2net-MNIST-with-growth-strat1:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model young --strat 1 --dataset MNIST --wandb disabled --run_name MNIST-with-growth-strat1

run-net2net-MNIST-without-growth_young-strat1:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model young --strat 1 --dataset MNIST --no_growth_for_baseline --wandb disabled --run_name MNIST-without-growth_young-strat1

run-net2net-MNIST-without-growth_adult-strat1:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model adult --strat 1 --dataset MNIST --no_growth_for_baseline  --wandb disabled --run_name MNIST-without-growth_adult-strat1

# Strat 2
run-net2net-MNIST-with-growth-strat2:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model young --strat 2 --dataset MNIST --wandb disabled --run_name MNIST-with-growth-strat2

run-net2net-MNIST-without-growth_young-strat2:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model young --strat 2 --dataset MNIST --no_growth_for_baseline --wandb disabled --run_name MNIST-without-growth_young-strat2

run-net2net-MNIST-without-growth_adult-strat2:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model adult --strat 2 --dataset MNIST --no_growth_for_baseline  --wandb disabled --run_name MNIST-without-growth_adult-strat2

# Strat 3
run-net2net-MNIST-with-growth-strat3:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model young --strat 3 --dataset MNIST --wandb disabled --run_name MNIST-with-growth-strat3

run-net2net-MNIST-without-growth_young-strat3:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model young --strat 3 --dataset MNIST --no_growth_for_baseline --wandb disabled --run_name MNIST-without-growth_young-strat3

run-net2net-MNIST-without-growth_adult-strat3:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model adult --strat 3 --dataset MNIST --no_growth_for_baseline  --wandb disabled --run_name MNIST-without-growth_adult-strat3

# Feature Transfer
run-net2net-MNIST-with-growth-featuretransfer-strat1:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train_knowledge_transfer --model young --strat 1 --dataset MNIST --wandb disabled --run_name Net2WiderNet_pretrained
run-net2net-MNIST-adult-baseline-featuretransfer-strat1:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train_knowledge_transfer --model adult --strat 1 --no_pretrain_for_baseline --dataset MNIST --wandb disabled --run_name Training_from_scratch



#==== Fashion MNIST ====#
# Strat 1
run-net2net-FashionMNIST-with-growth-strat1:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model young --strat 1 --dataset FASHION_MNIST --wandb disabled --run_name FASHION-with-growth-strat1

run-net2net-FashionMNIST-without-growth_young-strat1:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model young --strat 1 --dataset FASHION_MNIST --no_growth_for_baseline --wandb disabled --run_name FASHION-without-growth-young-strat1

run-net2net-FashionMNIST-without-growth_adult-strat1:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train --model adult --strat 1 --dataset FASHION_MNIST --no_growth_for_baseline --wandb disabled --run_name FASHION-without-growth-adult-strat1

# Knowledge Transfer
# Strategy 1 (Wider)
run-net2net-FashionMNIST-with-growth-featuretransfer-strat1:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train_knowledge_transfer --model young --strat 1 --dataset FASHION_MNIST --wandb disabled --run_name Net2WiderNet_pretrained_Fashion
run-net2net-FashionMNIST-adult-baseline-featuretransfer-strat1:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train_knowledge_transfer --model adult --strat 1 --no_pretrain_for_baseline --dataset FASHION_MNIST --wandb disabled --run_name Training_from_scratch_Fashion
# Strategy 2 (Wider + Deeper)
run-net2net-FashionMNIST-with-growth-featuretransfer-strat2:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train_knowledge_transfer --model young --strat 2 --dataset FASHION_MNIST --wandb disabled --run_name Net2WiderNet_pretrained_Fashion-strat2
run-net2net-FashionMNIST-adult-baseline-featuretransfer-strat2:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train_knowledge_transfer --model adult --strat 2 --no_pretrain_for_baseline --dataset FASHION_MNIST --wandb disabled --run_name Training_from_scratch_Fashion-strat2
# Strategy 3 (Wider + BN)
run-net2net-FashionMNIST-with-growth-featuretransfer-strat3:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train_knowledge_transfer --model young --strat 3 --dataset FASHION_MNIST --wandb online --run_name Net2WiderNet_pretrained_Fashion-strat3
run-net2net-FashionMNIST-adult-baseline-featuretransfer-strat3:
	$(PYTHON) -m scripts.net2net.net2net_MNIST.train_knowledge_transfer --model adult --strat 3 --no_pretrain_for_baseline --dataset FASHION_MNIST --wandb online --run_name Training_from_scratch_Fashion-strat3






###########################
### Sb Recipe ASR TIMIT ###
###########################

# MAC version with MPS
run-recipe-asr-timit-mac:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) -m scripts.speech.TIMIT.TIMIT_ASR_sb_recipe.train \
	scripts/speech/TIMIT/TIMIT_ASR_sb_recipe/hparams/train.yaml \
	--data_folder data/raw/TIMIT \
	--device=mps \
	--num_workers=0

# Normal Cuda version #
run-recipe-asr-timit:
	$(PYTHON) -m scripts.speech.TIMIT.TIMIT_ASR_sb_recipe.train \
	scripts/speech/TIMIT/TIMIT_ASR_sb_recipe/hparams/train.yaml \
	--data_folder data/raw/TIMIT \
	--jit

#####################
### Growing TIMIT ###
#####################

run-growing-TIMIT-adult-baseline:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) -m scripts.speech.TIMIT.TIMIT_Growing_Net2Net.train \
	scripts/speech/TIMIT/TIMIT_Growing_Net2Net/hparams/adult.yaml \
	--data_folder data/raw/TIMIT \
	--device=mps \
	--num_workers=0

run-growing-TIMIT-adult-baseline-colab:
	$(PYTHON) -m scripts.speech.TIMIT.TIMIT_Growing_Net2Net.train \
	scripts/speech/TIMIT/TIMIT_Growing_Net2Net/hparams/adult.yaml \
	--data_folder data/raw/TIMIT \
	--wandb_mode disabled

run-growing-TIMIT-young-baseline:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) -m scripts.speech.TIMIT.TIMIT_Growing_Net2Net.train \
	scripts/speech/TIMIT/TIMIT_Growing_Net2Net/hparams/young.yaml \
	--data_folder data/raw/TIMIT \
	--device=mps \
	--num_workers=0

run-growing-TIMIT-young-baseline-colab:
	$(PYTHON) -m scripts.speech.TIMIT.TIMIT_Growing_Net2Net.train \
	scripts/speech/TIMIT/TIMIT_Growing_Net2Net/hparams/young.yaml \
	--data_folder data/raw/TIMIT \
	--wandb_mode disabled

run-growing-TIMIT-young-growth:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) -m scripts.speech.TIMIT.TIMIT_Growing_Net2Net.train \
	scripts/speech/TIMIT/TIMIT_Growing_Net2Net/hparams/growing.yaml \
	--data_folder data/raw/TIMIT \
	--device=mps \
	--num_workers=0

run-growing-TIMIT-young-growth-colab:
	$(PYTHON) -m scripts.speech.TIMIT.TIMIT_Growing_Net2Net.train \
	scripts/speech/TIMIT/TIMIT_Growing_Net2Net/hparams/growing.yaml \
	--data_folder data/raw/TIMIT \
	--run_name growing-TIMIT-young-w-growth-at-5-TIMIT_Growing_young_at_5_noise_0_05

clear:
	trash results/TIMIT_Growing_young/

