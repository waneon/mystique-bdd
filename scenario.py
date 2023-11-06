import json
from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import shutil
import re

parser = ArgumentParser()
parser.add_argument("config")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = json.load(f)

scenario_name = config["scenario-name"]
crop_path = config["crop-path"]
output_root_path = config["output-root-path"]
windows = config["windows"]

out_img_path = Path(output_root_path, scenario_name, "images")
out_img_path.mkdir(parents=True, exist_ok=True)
ann = []

pattern = re.compile("scenario_\d*-(.*)")
global_id = 0

with ThreadPoolExecutor(100) as exe:
    for window in windows:
        window_path = Path(crop_path, window)

        with open(Path(window_path, "ann.json"), "r") as f:
            window_ann = json.load(f)

        for data in window_ann:
            image = data["image"]
            category = data["category"]

            foo = pattern.findall(image)[0]
            new_name = f"scenario_{global_id:08}-{foo}"

            ann.append({"image": new_name, "category": category})
            global_id += 1
            exe.submit(shutil.copy, Path(window_path, "images", image), Path(out_img_path, new_name))


with open(Path(output_root_path, scenario_name, "ann.json"), "w") as f:
    json.dump(ann, f, indent=2)
