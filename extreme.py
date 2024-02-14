import json
from pathlib import Path
from argparse import ArgumentParser
import re
import random
from PIL import Image
import torch
import copy

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import numpy as np
import torchvision
from tqdm import tqdm

DEVICE = "cuda:0"

parser = ArgumentParser()
parser.add_argument("config")
args = parser.parse_args()


class MyDataset(Dataset):
    def __init__(self, path: Path):
        with open(Path(path, "ann.json"), "r") as f:
            ann = json.load(f)

        self.len = len(ann)
        self.x = []
        for x in ann:
            with Image.open(Path(path, "images", x["image"])) as img:
                self.x.append(img.copy())
        self.y = [x["category"] for x in ann]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        t = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return t(self.x[index]), self.y[index]

def train(model, path: Path, optim):
    dataset = MyDataset(path)
    dataloader = DataLoader(dataset, batch_size=32)
    loss_fn = torch.nn.CrossEntropyLoss()

    accs = []
    model.train()
    for x, y in dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        y_predict = model(x)
        accs.extend((y_predict.argmax(dim=1) == y).cpu())
        loss = loss_fn(y_predict, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

    return np.mean(accs)


def eval(model, path: Path):
    dataset = MyDataset(path)
    dataloader = DataLoader(dataset, batch_size=32)

    accs = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_predict = model(x)
            accs.extend((y_predict.argmax(dim=1) == y).cpu())

    return np.mean(accs)


with open(args.config, "r") as f:
    config = json.load(f)

scenario_name = config["scenario-name"]
crop_path = config["crop-path"]
output_root_path = config["output-root-path"]
windows = list(Path(crop_path).iterdir())
num_windows: int = config["num-windows"]

out_img_path = Path(output_root_path, scenario_name, "images")
out_img_path.mkdir(parents=True, exist_ok=True)
outputs = []

t = random.choice(windows)
outputs.append(t)
windows.remove(t)

pattern = re.compile("scenario_\d*-(.*)")
global_id = 0

model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 9, bias=False)
model = model.to(DEVICE)

acc_prev = eval(model, outputs[0])
print(outputs[-1], acc_prev)
for _ in range(num_windows - 1):
    best_diff = 0.0
    best_path = None
    best_model = None
    best_acc = 0.0
    for window in tqdm(random.sample(windows, 10)):
        model_cloned = copy.deepcopy(model)
        opt = torch.optim.Adam(model_cloned.parameters(), lr=0.1)

        train(model_cloned, window, opt)
        acc_cur = eval(model_cloned, window)

        if best_diff < abs(acc_prev - acc_cur):
            best_diff = abs(acc_prev - acc_cur)
            best_path = window
            best_model = model_cloned
            best_acc = acc_cur

    acc_prev = best_acc
    print(outputs[-1], acc_prev)

    model = best_model
    outputs.append(best_path)
    windows.remove(best_path)
