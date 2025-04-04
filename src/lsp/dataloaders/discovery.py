"""Dataset discovery utilities for finding and loading datasets."""

import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Iterator, Type, Set

from lsp.core.data_loader import DataLoader
from lsp.dataloaders.alea import AleaDataLoader
from lsp.dataloaders.multilegal import MultiLegalSBDLoader


def get_data_directories(base_dir: Optional[str] = None) -> Dict[str, Path]:
    """Get all data directories from the project.
    
    Args:
        base_dir: Optional base directory to search from.
                If None, tries to find the project root.
    
    Returns:
        Dictionary of dataset name to path
    """
    # If base_dir not provided, try to find project root
    if base_dir is None:
        # Try to find the project root by looking for data/
        current_dir = Path.cwd()
        while current_dir != current_dir.parent:
            if (current_dir / "data").exists():
                base_dir = str(current_dir)
                break
            current_dir = current_dir.parent
    
    if base_dir is None:
        raise ValueError("Could not find project root (directory with 'data/')")
    
    root_dir = Path(base_dir)
    
    # Define the dataset directories as specified in CLAUDE.md
    return {
        "alea": root_dir / "data" / "alea-legal-benchmark",
        "multilegal": root_dir / "data" / "MultiLegalSBD"
    }


def discover_datasets(base_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Discover all available datasets.
    
    Args:
        base_dir: Optional base directory to search from.
                If None, tries to find the project root.
    
    Returns:
        Dictionary of dataset ID to dataset information
    """
    data_dirs = get_data_directories(base_dir)
    datasets = {}
    
    # Discover ALEA datasets
    if data_dirs["alea"].exists():
        jsonl_files = list(data_dirs["alea"].glob("*.jsonl"))
        for file_path in jsonl_files:
            dataset_id = f"alea_{file_path.stem}"
            datasets[dataset_id] = {
                "type": "alea",
                "name": f"ALEA {file_path.stem}",
                "path": str(file_path),
                "format": "<|sentence|> annotation",
                "loader_class": AleaDataLoader
            }
    
    # Discover MultiLegalSBD datasets
    if data_dirs["multilegal"].exists():
        jsonl_files = list(data_dirs["multilegal"].glob("*.jsonl"))
        for file_path in jsonl_files:
            # Extract meaningful name from filename (e.g., CD_scotus.jsonl -> scotus)
            name_parts = file_path.stem.split("_")
            if len(name_parts) > 1:
                name = "_".join(name_parts[1:])
            else:
                name = file_path.stem
                
            dataset_id = f"multilegal_{name.lower()}"
            datasets[dataset_id] = {
                "type": "multilegal",
                "name": f"MultiLegalSBD {name}",
                "path": str(file_path),
                "format": "span annotation",
                "loader_class": MultiLegalSBDLoader
            }
    
    return datasets


def load_dataset(dataset_info: Dict[str, Any], **kwargs) -> DataLoader:
    """Load a dataset based on its information.
    
    Args:
        dataset_info: Dataset information from discover_datasets
        **kwargs: Additional arguments to pass to the loader
    
    Returns:
        Loaded data loader instance
    """
    loader_class = dataset_info["loader_class"]
    name = dataset_info.get("name", "")
    
    # Create loader
    loader = loader_class(name=dataset_info.get("id", name))
    
    # Load dataset
    loader.load(dataset_info["path"], **kwargs)
    
    return loader


def load_all_datasets(base_dir: Optional[str] = None, 
                     dataset_types: Optional[List[str]] = None,
                     limit: Optional[int] = None,
                     **kwargs) -> Dict[str, DataLoader]:
    """Discover and load all datasets.
    
    Args:
        base_dir: Optional base directory to search from
        dataset_types: Optional list of dataset types to load ("alea", "multilegal")
        limit: Optional maximum number of examples to load per dataset
        **kwargs: Additional arguments to pass to each loader
    
    Returns:
        Dictionary of dataset ID to loaded data loader
    """
    # Discover available datasets
    datasets_info = discover_datasets(base_dir)
    loaded_datasets = {}
    
    # Filter by dataset type if specified
    if dataset_types:
        datasets_info = {
            k: v for k, v in datasets_info.items() 
            if v["type"] in dataset_types
        }
    
    # Load each dataset
    for dataset_id, dataset_info in datasets_info.items():
        try:
            print(f"Loading dataset: {dataset_info['name']} ({dataset_id})")
            
            # Add dataset ID and limit if provided
            dataset_info["id"] = dataset_id
            if limit:
                kwargs["limit"] = limit
                
            # Load dataset
            loaded_datasets[dataset_id] = load_dataset(dataset_info, **kwargs)
            
            print(f"  Loaded {len(loaded_datasets[dataset_id])} examples")
            
        except Exception as e:
            print(f"  Failed to load dataset {dataset_id}: {e}")
    
    return loaded_datasets


def get_dataset_registry() -> Dict[str, Type[DataLoader]]:
    """Get a registry of all available dataset loader classes.
    
    Returns:
        Dictionary of dataset type to loader class
    """
    return {
        "alea": AleaDataLoader,
        "multilegal": MultiLegalSBDLoader
    }


def get_dataset_extensions() -> Dict[str, List[str]]:
    """Get file extensions for each dataset type.
    
    Returns:
        Dictionary of dataset type to list of extensions
    """
    return {
        "alea": [".jsonl"],
        "multilegal": [".jsonl"]
    }


def dataset_type_from_path(path: str) -> Optional[str]:
    """Try to determine dataset type from path.
    
    Args:
        path: Path to the dataset file
    
    Returns:
        Dataset type or None if unknown
    """
    path_obj = Path(path)
    
    # Check file extension
    if not path_obj.exists() or not path_obj.is_file():
        return None
        
    # Try to determine by path components
    path_str = str(path_obj).lower()
    
    if "alea-legal-benchmark" in path_str:
        return "alea"
    elif "multilegal" in path_str or "multilegalsbd" in path_str:
        return "multilegal"
    
    # Try to determine by file content (peek at first line)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if "<|sentence|>" in first_line:
                return "alea"
            elif '"spans":' in first_line and '"label":"Sentence"' in first_line:
                return "multilegal"
    except:
        pass
    
    return None


def create_loader_for_path(path: str, **kwargs) -> Optional[DataLoader]:
    """Create appropriate loader for a dataset file.
    
    Args:
        path: Path to the dataset file
        **kwargs: Additional arguments to pass to the loader
    
    Returns:
        Loaded data loader or None if no appropriate loader found
    """
    dataset_type = dataset_type_from_path(path)
    if not dataset_type:
        return None
        
    registry = get_dataset_registry()
    if dataset_type not in registry:
        return None
        
    loader_class = registry[dataset_type]
    
    # Extract a name from the path
    name = Path(path).stem.lower()
    loader = loader_class(name=f"{dataset_type}_{name}")
    
    # Load the dataset
    loader.load(path, **kwargs)
    
    return loader