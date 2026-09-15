import os
import re
import glob
import numpy as np
from ogbench.utils import load_dataset

# kept free of `utils` imports: utils/buffer.py imports envs, so importing utils here would be circular

_SIZE_TOKEN = re.compile(r'^\d+[mb]$')    # dataset size token, e.g. '100m' in 'cube-quadruple-play-100m-v0'


def is_sharded_ogbench_dataset(dataset_name: str) -> bool:
    """large OGBench datasets (100M/1B) are directories of shards named with a size token"""
    return any(_SIZE_TOKEN.match(token) for token in dataset_name.split('-'))


def ogbench_env_dataset_name(dataset_name: str) -> str:
    """strip the size token so ogbench can derive the env id ('cube-quadruple-play-100m-v0' -> 'cube-quadruple-play-v0')"""
    return '-'.join(token for token in dataset_name.split('-') if not _SIZE_TOKEN.match(token))


def list_ogbench_shards(dataset_dir: str) -> list:
    """sorted training shards in a dataset directory (validation shards excluded), as in horizon-reduction"""
    shards = [f for f in sorted(glob.glob(os.path.join(dataset_dir, '*.npz'))) if not f.endswith('-val.npz')]
    if not shards:
        raise FileNotFoundError(f"No training shards (*.npz) found in {dataset_dir}. Download the sharded dataset into this directory first.")
    return shards


def load_ogbench_shard(path: str, dataset_name: str) -> dict:
    """load one shard in the compact format expected by Buffer.load_offline_dataset"""
    if 'singletask' in dataset_name or 'oraclerep' in dataset_name:
        raise NotImplementedError(f"sharded loading of '{dataset_name}' is not supported (singletask/oraclerep datasets need extra processing)")
    ob_dtype = np.uint8 if 'visual' in dataset_name else np.float32
    return load_dataset(path, ob_dtype=ob_dtype, action_dtype=np.float32, compact_dataset=True)
