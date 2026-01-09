#!/usr/bin/env fish
# ============================================================
# gsplat 环境配置脚本 (Fish Shell)
# 基于已有的3DGS环境配置,保持 Python 3.10.12, PyTorch 2.4.1, CUDA 12.1
# gsplat: CUDA accelerated rasterization of gaussian splatting
# GitHub: https://github.com/nerfstudio-project/gsplat
# ============================================================

# 环境名称
set ENV_NAME "gsplat"

echo "=========================================="
echo "开始配置 gsplat 环境"
echo "=========================================="

# 创建conda环境
conda create -y -n $ENV_NAME python=3.10.12
conda activate $ENV_NAME

# ============================================================
# PyTorch + CUDA 12.1 (保持与你的设置一致)
# ============================================================
echo "安装 PyTorch + CUDA 12.1..."
conda install pytorch==2.4.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --yes
conda install cuda-toolkit -c nvidia/label/cuda-12.1.0 --yes

# 解决 undefined symbol: iJIT_NotifyEvent
conda install mkl==2023.1.0 mkl-include -c conda-forge --yes

# 解决 cannot find -lcudart
conda install cuda-cudart=12.1.55 -c nvidia/label/cuda-12.1.0 --yes

# ============================================================
# gsplat 核心依赖
# ============================================================
echo "安装 gsplat 核心依赖..."
pip install ninja \
    numpy \
    jaxtyping \
    "rich>=12"

# ============================================================
# 安装 gsplat
# ============================================================
echo "=========================================="
echo "安装 gsplat..."
echo "=========================================="

# 方法1: 从PyPI安装 (推荐,JIT编译)
# pip install gsplat

# 或者方法2: 从源码安装 (如果需要最新开发版本)
# git clone --recursive https://github.com/nerfstudio-project/gsplat.git
# cd gsplat
pip install -e . --no-build-isolation 

# ============================================================
# 安装 examples 依赖 (如果需要运行示例)
# ============================================================
echo "=========================================="
echo "安装 examples 依赖 (可选)..."
echo "=========================================="


# 安装 examples/requirements.txt 中的依赖
pip install -r examples/requirements.txt  --no-build-isolation 

# COLMAP 相关 (用于处理数据集)
pip install plyfile \
    imageio

# 可视化工具 (可选)
# git clone https://github.com/nerfstudio-project/nerfview.git
# cd nerfview
# pip install -e .

# ============================================================
# 验证安装
# ============================================================
echo "=========================================="
echo "验证安装..."
echo "=========================================="

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import gsplat; print(f'gsplat imported successfully')"

echo "=========================================="
echo "gsplat 环境配置完成!"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  conda activate $ENV_NAME"
echo ""
echo "运行示例 (需要先克隆仓库):"
echo "  cd examples"
echo "  python simple_trainer.py default --data_dir <path_to_data>"
echo ""
echo "或者使用预编译wheel (可选,适用于特定PyTorch+CUDA版本):"
echo "  pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu121"
echo "=========================================="