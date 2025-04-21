from hyperpyyaml import load_hyperpyyaml
import os


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


yaml_path_adult = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "hparams/adult.yaml"
)
yaml_path_young = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "hparams/young.yaml"
)

with open(yaml_path_adult) as f:
    hparams_adult = load_hyperpyyaml(f, overrides={"data_folder": ""})
with open(yaml_path_young) as f:
    hparams_young = load_hyperpyyaml(f, overrides={"data_folder": ""})

n_params_young = count_params(hparams_young["model"])
n_params_adult = count_params(hparams_adult["model"])

print("Young model params:", n_params_young)
print(hparams_young["model"])
print("Adult model params:", n_params_adult)
print(hparams_adult["model"])
print(hparams_adult["output"])
print("Difference:", n_params_adult - n_params_young)
