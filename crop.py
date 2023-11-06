from argparse import ArgumentParser
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
import re

import toml
from PIL import Image

LABEL_LIST = {
    "car": 0,
    "traffic sign": 1,
    "traffic light": 2,
    "person": 3,
    "truck": 4,
    "bus": 5,
    "bike": 6,
    "rider": 7,
    "motor": 8,
    "train": 9,
}

parser = ArgumentParser()
parser.add_argument("config")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = toml.load(f)

original_path: str = config["original-path"]
crop_root_path: str = config["crop-root-path"]
fps = config["fps"]
sec = config["sec"]
images_per_window: int = fps * sec


def crop(img_path: Path, bbox: Tuple[int, int, int, int], save_path: Path):
    x, y, x2, y2 = bbox
    w = x2 - x
    h = y2 - y

    margin = 0.5
    new_width = max((2.0 * margin + 1.0) * w, (2.0 * margin + 1.0) * h)

    offset_x = (new_width - w) / 2.0
    offset_y = (new_width - h) / 2.0

    x, y, w, h = int(x - offset_x), int(y - offset_y), int(new_width), int(new_width)

    with Image.open(img_path) as img:
        cropped = img.crop((x, y, x + w, y + h))
        cropped.save(save_path)


pattern = re.compile("(.*)-(.*)-(.*)-(.*)")
with ThreadPoolExecutor(100) as exe:
    global_cnt = 0
    for scene_idx, scene in enumerate(Path(original_path, "filtered").glob("*json")):
        weather, location, time, dist_type = pattern.findall(scene.stem)[0]

        with open(scene, "r") as f:
            data = json.load(f)

        cnt = 0
        window_idx = 0
        ann = []

        out_dir = Path(
            crop_root_path,
            f"crop-fps{fps}-sec{sec}",
            f"{scene.stem}-window_{0:03}",
        )
        out_dir.mkdir(exist_ok=True, parents=True)
        Path(out_dir, "images").mkdir(exist_ok=True, parents=True)
        for img in data:
            name = img["name"]

            for obj in img["labels"]:
                if cnt >= images_per_window:
                    cnt = 0
                    window_idx += 1

                    with open(Path(out_dir, "ann.json"), "w") as f:
                        json.dump(ann, f, indent=2)

                    out_dir = Path(
                        crop_root_path,
                        f"crop-fps{fps}-sec{sec}",
                        f"{scene.stem}-window_{window_idx:03}",
                    )
                    out_dir.mkdir(exist_ok=True, parents=True)
                    Path(out_dir, "images").mkdir(exist_ok=True, parents=True)

                    ann = []

                category = LABEL_LIST[obj["category"]]
                bbox = obj["box2d"].values()

                new_name = f"scenario_{global_cnt:08}-scene_{scene_idx}-window_{window_idx}-weather_{weather}-location_{location}-time_{time}-dist_type_{dist_type}-label_{category}.jpg"

                from_path = Path(original_path, "images", "100k", "train", name)
                ann.append(
                    {
                        "image": new_name,
                        "category": category,
                    }
                )

                exe.submit(crop, from_path, bbox, Path(out_dir, "images", new_name))

                cnt += 1
                global_cnt += 1

        if cnt > 0:
            with open(Path(out_dir, "ann.json"), "w") as f:
                json.dump(ann, f, indent=2)
