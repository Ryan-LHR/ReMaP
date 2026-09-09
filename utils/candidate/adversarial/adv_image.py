from tqdm import tqdm
from torchvision.datasets import ImageFolder

from utils import get_device
from utils.candidate.adversarial.adv_image_utils import _normalize_batch, _denormalize_batch
from utils.candidate.adversarial.imagenet_mix import make_mixed_im100_dataset
from utils.datasets import CustomTensorDataset
from utils.candidate.adversarial.adv_image_utils import *
from utils.candidate.adversarial.imagenet_utils import \
    save_adv_batch_imagenet_style, save_adv_batch_imagenet_style_pil, create_temp_dataloader

TENSOR_DATASETS = ['mnist', 'fashion_mnist', 'cifar_10', 'svhn']

def generate_adv_image(dataset_name, model_name, attack_types, model, dataloader, path_to_data):
    """
    Generate adversarial samples for candidate set
    """
    "(1) Setup"
    attack_tag = "+".join(attack_types) if isinstance(attack_types, (list, tuple)) else str(attack_types)
    device = get_device()

    cfg = DATASET_ATTACK_CFG[dataset_name]
    mean = cfg.get("mean", None)
    std = cfg.get("std", None)

    if mean is not None and std is not None:
        wrapped_model = ModelWrapper(model, mean, std).to(device)
    else:
        wrapped_model = model
    wrapped_model.eval()

    "(2) Build attacker"
    attacker = CompositeAttacker(attack_types, mean, std, dataset_name, wrapped_model, seed=0)

    "(3) Attack"
    input_probe, _ = next(iter(dataloader))

    print(f"The value range of first batch inputs: "
          f"[{input_probe.min().item()}, {input_probe.max().item()}]")

    if dataset_name in TENSOR_DATASETS:
        all_inputs, all_labels = [], []
        for inputs, labels in tqdm(dataloader,
                                   desc="Generating adversarial inputs",
                                   total=len(dataloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs_adv = attacker(inputs, labels)

            all_inputs.append(inputs_adv)
            all_labels.append(labels)

        print(f"The value range of last batch adversarial inputs: "
              f"[{inputs_adv.min().item()}, {inputs_adv.max().item()}]")
        inputs_tensor = torch.cat(all_inputs,  dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        "(4) Get Dataset"
        save_adversarial_tensor_dataset(dataset_name, model_name, inputs_tensor, labels_tensor, path_to_data)

    elif dataset_name in ['imagenet_100']:
        dataloader = create_temp_dataloader(path_to_data, dataset_name)

        out_root = os.path.join(
            path_to_data,
            dataset_name,
            "adv",
            f"{model_name}_{attack_tag}",
            "val"
        )

        for inputs, labels, paths in tqdm(
                dataloader,
                desc="Generating adversarial inputs",
                total=len(dataloader)
        ):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            inputs_adv = attacker(inputs, labels)
            inputs_adv = _denormalize_batch(inputs_adv, wrapped_model.mean, wrapped_model.std)

            save_adv_batch_imagenet_style_pil(
                inputs_adv=inputs_adv,
                paths=paths,
                out_root=out_root,
                skip_if_exists=True,
            )
            print(f"The value range of last batch adversarial inputs: "
                  f"[{inputs_adv.min().item()}, {inputs_adv.max().item()}]")


def load_adversarial_im100(path_to_data, model_name, attack_types, mix):

    attack_tag = "+".join(attack_types) if isinstance(attack_types, (list, tuple)) else str(attack_types)
    if model_name == "IM100Test-deit_base_patch16_224":
        model_name = "IM100-FastViT_S12"
    adv_dir = os.path.join(
        path_to_data,
        "imagenet_100",
        "adv",
        f"{model_name}_{attack_tag}",
        "val"
    )

    if mix:
        attack_tag += '_mix'
        val_dir = os.path.join(path_to_data, f'imagenet_100/raw/val')
        save_dir = os.path.join(
            path_to_data,
            "imagenet_100",
            "adv",
            f"{model_name}_{attack_tag}",
            "val"
        )
        if not os.path.exists(save_dir):
            make_mixed_im100_dataset(
                val_root=val_dir,
                adv_root=adv_dir,
                out_root=save_dir,
                seed=42,
            )
    else:
        save_dir = adv_dir

    print(f"===> Loading processed Images from: {save_dir}")
    dataset = create_dataset(
        root=save_dir,
        name='',
        split='validation',
        download=False,
        load_bytes=False,
        class_map='',
    )

    return dataset