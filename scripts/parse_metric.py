from itertools import product
import json
from pathlib import Path
from pandas import DataFrame as Df
import pandas as pd

gs_output_root = Path("/mnt/cviss/Matt/GS-Output")
items = []
for method in ["GSplat-default", "GSplat-mcmc", "GSplat-resize-default", "GSplat-resize-mcmc"]:
    root = gs_output_root / method
    for training_time_file in root.rglob("training_time.json"):
        rel_path = training_time_file.relative_to(root)
        stats_rel = rel_path.parent
        scene_rel = stats_rel.parent
        dataset_name = scene_rel.parts[0]
        scene_name = Path(*scene_rel.parts[1:])
        scene_name, downsampling = str(scene_name).rsplit('_', 1)

        dataset_dir = root / dataset_name
        scene_dir = root / scene_rel
        stats_dir = scene_dir / "stats"

        training_time_file = stats_dir / "training_time.json"
        
        item = {
            "dataset": dataset_name,
            "scene": scene_name,
            "method": method,
            "downsampling": downsampling,
        }
        with open(training_time_file, 'r') as f:
            training_time_data = json.load(f)
        item["_times"] = training_time_data["train_times"]
        item["_train_render_times"] = training_time_data["train_render_times"]  
        item["_train_optimal_times"] = training_time_data["train_optimal_times"]
        for stage, step in product(["val", "train"], [6999, 29999]):
            eval_file = stats_dir / f"{stage}_step{step}.json"
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
            step = step + 1
            step = step // 1000
            if stage == "val":
                stage = "test"
            item.update({
                f"psnr_{step}k_{stage}": eval_data["psnr"],
                f"ssim_{step}k_{stage}": eval_data["ssim"],    
                f"lpips_{step}k_{stage}": eval_data["lpips"],
            })
            if stage == "test" and step == 30:
                item["gs_number"] = eval_data["num_GS"]
        items.append(item)

df = pd.DataFrame(items)
df["gpu"] = "RTX 4090"
df["resolution"] = None
df = df[["method", "dataset", "scene", "gpu", "downsampling", "psnr_7k_train", "ssim_7k_train", "lpips_7k_train", "psnr_7k_test", "ssim_7k_test", "lpips_7k_test", "psnr_30k_train", "ssim_30k_train", "lpips_30k_train", "psnr_30k_test", "ssim_30k_test", "lpips_30k_test", "resolution", "gs_number", "_times", "_train_render_times", "_train_optimal_times"]]
df.to_csv("results_summary.csv", index=False, sep='\t')
