from itertools import product
import os
from pathlib import Path
import pty
import select
import subprocess

def run_with_live_output(cmd):
    """运行命令并实时显示输出，正确处理 tqdm"""
    print(f"Running command: \n{cmd}")
    master_fd, slave_fd = pty.openpty()
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True
    )
    os.close(slave_fd)
    
    output_lines = []
    
    while True:
        ready, _, _ = select.select([master_fd], [], [], 0.1)
        if ready:
            try:
                data = os.read(master_fd, 1024).decode('utf-8', errors='replace')
                if data:
                    print(data, end='', flush=True)
                    output_lines.append(data)
            except OSError:
                break
        
        if process.poll() is not None:
            # 读取剩余输出
            while True:
                try:
                    data = os.read(master_fd, 1024).decode('utf-8', errors='replace')
                    if not data:
                        break
                    print(data, end='', flush=True)
                    output_lines.append(data)
                except OSError:
                    break
            break
    
    os.close(master_fd)
    return process.returncode, ''.join(output_lines)

def get_train_cmd(input, output, factor=None, strategy="default"):

    # 路径配置
    image_root = f"/home/matt/cviss/Matt/Dataset/{input}"  # Blendswap/Render/pick/13078_toad
    output_base_dir = f"/home/matt/cviss/Matt/GS-Output"
    output_full_dir = f"{output_base_dir}/GSplat-resize{('-' + strategy) if strategy else ''}/{output}"  # Blendswap/Render/pick/13078_toad'strategy}/{output}"  # pick/13078_toad

    cmd = (
        f"OMP_NUM_THREADS=4 "
        f"CUDA_VISIBLE_DEVICES=0 "
        f"python examples/simple_trainer.py "
        f"{strategy} "
        f"--data_dir {image_root} "
        f"--result_dir {output_full_dir} "
        f"--eval_steps -1 "
        f"--disable_viewer "
        f"--data_factor {factor} "

    )
    # ============================================================
    # 评估命令
    # ============================================================
    eval_cmd = (
        f"OMP_NUM_THREADS=4 "
        f"CUDA_VISIBLE_DEVICES=0 "
        f"python examples/simple_trainer.py "
        f"{strategy} "
        f"--data_dir {image_root} "
        f"--result_dir {output_full_dir} "
        f"--disable_viewer "
        f"--ckpt ckpt_6999_rank0.pt ckpt_29999_rank0.pt"
    )
    # eval_cmd = None
    return cmd, eval_cmd


if __name__ == "__main__":
    # for factor, strategy in product([2], ["mcmc", "default"]):
    #     # 示例用法
    #     cmd, eval_cmd = get_train_cmd(input="Rogers/Tower_0529", output=f"Rogers/Tower_0529_{factor}", factor=factor, strategy=strategy)

    #     run_with_live_output(cmd)
    #     run_with_live_output(eval_cmd)
    
    
    # MipNerf-360 dataset
    dataset_root = Path("/mnt/cviss/Matt/Dataset")
    root = dataset_root / "Mip-NeRF360/360_v2/"
    for scene_dir in root.iterdir():
        if not scene_dir.is_dir():
            continue
        input_path = f"Mip-NeRF360/360_v2/{scene_dir.name}"
        for factor, strategy in product([2], ["mcmc", "default"]):
            output_path = f"Mip-NeRF360/360_v2/{scene_dir.name}_{factor}"
            cmd, eval_cmd = get_train_cmd(input=input_path, output=output_path, factor=factor, strategy=strategy)
            run_with_live_output(cmd)
            run_with_live_output(eval_cmd)
    
    cmd, eval_cmd = get_train_cmd(input="Blendswap/Render/pick/13078_toad", output="Blendswap/pick/13078_toad", image_dir=None)

    run_with_live_output(cmd)
    run_with_live_output(eval_cmd)