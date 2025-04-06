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

# Commands
run-basic-MNIST:
	$(PYTHON) -m scripts.net2net.basic_MNIST.train