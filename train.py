import os
import argparse
import shutil
import torch
import tqdm
import time

import utils
from data_loading import get_all_data_loaders
from transforms import get_transforms
from model_manager import ModelManager
from test import ModelTester

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/abo.yaml")
parser.add_argument("--id", type=str, default="none", help="experiment ID")
parser.add_argument("--output_path", type=str, default="./outputs")
parser.add_argument("--processed_dir_name", type=str, default="processed")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--profile", action="store_true")
parsed = parser.parse_args()

if parsed.id != "none":
    experiment_name = parsed.id
else:
    experiment_name = os.path.splitext(os.path.basename(parsed.config))[0]
print(f"Experiment {experiment_name}")

# Create output directory and checkpoints directory (in output directory)
output_directory = os.path.join(parsed.output_path, experiment_name)
checkpoint_directory = os.path.join(output_directory, "checkpoints")
utils.mkdir(checkpoint_directory)

if parsed.resume:
    config = utils.get_config(os.path.join(output_directory, "config.yaml"))
else:
    config = utils.get_config(parsed.config)
    shutil.copy(parsed.config, os.path.join(output_directory, "config.yaml"))
data_config = config["data"]

# Use GPU if available
if not torch.cuda.is_available():
    device = torch.device("cpu")
    print("GPU not available, running on CPU")
else:
    device = torch.device("cuda")

pre_transform = get_transforms(
    list_transforms=data_config["pre_transforms_list"],
    transforms_config=data_config["transforms_config"],
    root=data_config["root"],
)
transform = get_transforms(
    list_transforms=data_config["transforms_list"],
    transforms_config=data_config["transforms_config"],
    root=data_config["root"],
)

loaders = get_all_data_loaders(
    data_config, transform, pre_transform, None, parsed.processed_dir_name
)

model_manager = ModelManager(config, len_train_loader=len(loaders["train"]))
model_manager = model_manager.to(device)

start_epoch = 0
if parsed.resume:
    start_epoch = model_manager.resume(checkpoint_directory)
if parsed.profile:
    model_manager.create_profiler(output_directory)
    model_manager.run_profiling_epoch(loaders["train"])
model_manager.create_tb_logger(output_directory)
model_manager.pick_shapes_for_logging(loaders, number_of_shapes=5)

for epoch in tqdm.tqdm(
    range(start_epoch, config["epochs"]), desc="Epochs: ", position=0, unit="ep"
):
    model_manager.run_epoch(loaders["train"], train=True, nest_bar=True)
    model_manager.log_loss(epoch, "train")

    model_manager.run_epoch(loaders["val"], train=False, nest_bar=True)
    model_manager.log_loss(epoch, "val")

    if (epoch + 1) % config["log_frequency"]["save_checkpoint"] == 0:
        model_manager.save(checkpoint_directory, epoch)

    if (epoch + 1) % config["log_frequency"]["log_renderings"] == 0:
        model_manager.log_generated_images(epoch, "train", n_variants=2)
        model_manager.log_generated_images(epoch, "val", n_variants=2)

time.sleep(3)  # otherwise last image not logged

tester = ModelTester(
    model_manager, loaders, config, device, output_directory, phase="val"
)
tester()  # run all tests on validation set
