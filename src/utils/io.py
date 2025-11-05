"""
I/O utilities for data caching and storage
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from typing import Any, Dict, Optional
import hashlib


def save_parquet(df: pd.DataFrame, filepath: Path, **kwargs):
    """
    Save DataFrame to parquet with compression

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filepath : Path
        Output filepath
    **kwargs : dict
        Additional arguments for to_parquet
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    default_kwargs = {'compression': 'snappy', 'index': True}
    default_kwargs.update(kwargs)

    df.to_parquet(filepath, **default_kwargs)
    print(f"Saved to {filepath} ({filepath.stat().st_size / 1024 / 1024:.2f} MB)")


def load_parquet(filepath: Path, **kwargs) -> pd.DataFrame:
    """
    Load DataFrame from parquet

    Parameters:
    -----------
    filepath : Path
        Input filepath
    **kwargs : dict
        Additional arguments for read_parquet

    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    return pd.read_parquet(filepath, **kwargs)


def save_pickle(obj: Any, filepath: Path):
    """
    Save object to pickle

    Parameters:
    -----------
    obj : Any
        Object to save
    filepath : Path
        Output filepath
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved to {filepath}")


def load_pickle(filepath: Path) -> Any:
    """
    Load object from pickle

    Parameters:
    -----------
    filepath : Path
        Input filepath

    Returns:
    --------
    Any
        Loaded object
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(data: Dict, filepath: Path, indent: int = 2):
    """
    Save dictionary to JSON

    Parameters:
    -----------
    data : dict
        Data to save
    filepath : Path
        Output filepath
    indent : int
        JSON indentation
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)

    print(f"Saved to {filepath}")


def load_json(filepath: Path) -> Dict:
    """
    Load dictionary from JSON

    Parameters:
    -----------
    filepath : Path
        Input filepath

    Returns:
    --------
    dict
        Loaded data
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r') as f:
        return json.load(f)


def get_cache_path(cache_dir: Path, key: str, ext: str = 'parquet') -> Path:
    """
    Generate cache filepath from key

    Parameters:
    -----------
    cache_dir : Path
        Cache directory
    key : str
        Cache key
    ext : str
        File extension

    Returns:
    --------
    Path
        Cache filepath
    """
    # Hash the key to create filename
    key_hash = hashlib.md5(key.encode()).hexdigest()[:16]
    return cache_dir / f"cache_{key_hash}.{ext}"


def cache_dataframe(
    df: pd.DataFrame,
    cache_dir: Path,
    cache_key: str,
    overwrite: bool = False
) -> Path:
    """
    Cache DataFrame to parquet

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to cache
    cache_dir : Path
        Cache directory
    cache_key : str
        Unique cache key
    overwrite : bool
        Overwrite existing cache

    Returns:
    --------
    Path
        Cache filepath
    """
    cache_path = get_cache_path(cache_dir, cache_key)

    if cache_path.exists() and not overwrite:
        print(f"Cache exists: {cache_path}")
        return cache_path

    save_parquet(df, cache_path)
    return cache_path


def load_cached_dataframe(
    cache_dir: Path,
    cache_key: str
) -> Optional[pd.DataFrame]:
    """
    Load cached DataFrame

    Parameters:
    -----------
    cache_dir : Path
        Cache directory
    cache_key : str
        Unique cache key

    Returns:
    --------
    pd.DataFrame or None
        Cached DataFrame if exists, else None
    """
    cache_path = get_cache_path(cache_dir, cache_key)

    if not cache_path.exists():
        return None

    print(f"Loading cache: {cache_path}")
    return load_parquet(cache_path)


def clear_cache(cache_dir: Path, pattern: str = "cache_*.parquet"):
    """
    Clear cache files matching pattern

    Parameters:
    -----------
    cache_dir : Path
        Cache directory
    pattern : str
        Glob pattern for cache files
    """
    cache_files = list(cache_dir.glob(pattern))

    for filepath in cache_files:
        filepath.unlink()
        print(f"Deleted: {filepath}")

    print(f"Cleared {len(cache_files)} cache files")


def save_model_checkpoint(
    model_state: Dict,
    filepath: Path,
    metadata: Optional[Dict] = None
):
    """
    Save model checkpoint with metadata

    Parameters:
    -----------
    model_state : dict
        Model state dictionary
    filepath : Path
        Output filepath
    metadata : dict, optional
        Additional metadata
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state': model_state,
        'metadata': metadata or {},
        'timestamp': pd.Timestamp.now().isoformat()
    }

    save_pickle(checkpoint, filepath)


def load_model_checkpoint(filepath: Path) -> Dict:
    """
    Load model checkpoint

    Parameters:
    -----------
    filepath : Path
        Checkpoint filepath

    Returns:
    --------
    dict
        Checkpoint dictionary
    """
    checkpoint = load_pickle(filepath)

    print(f"Loaded checkpoint from {filepath}")
    if 'metadata' in checkpoint:
        print(f"Metadata: {checkpoint['metadata']}")

    return checkpoint


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import tempfile

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())

    # Test DataFrame caching
    df = pd.DataFrame({
        'A': np.random.randn(1000),
        'B': np.random.randn(1000)
    })

    print("Testing DataFrame caching...")
    cache_path = cache_dataframe(df, temp_dir, "test_df")
    df_loaded = load_cached_dataframe(temp_dir, "test_df")

    assert df.equals(df_loaded), "DataFrames don't match!"
    print("✓ DataFrame caching works")

    # Clean up
    clear_cache(temp_dir)
    print(f"\nTemp dir: {temp_dir}")
