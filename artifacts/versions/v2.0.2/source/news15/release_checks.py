"""Allow an audited UI successor while retaining the same scientific release."""
import json
import re
from pathlib import Path

from .data import file_sha256
from .paths import PROJECT_DIR


def compatible_ui_pointer(expected_pointer_hash, expected_model_hash, project_dir=PROJECT_DIR):
    root=Path(project_dir)
    pointer=root/'artifacts/active_version.json'
    if file_sha256(root/'artifacts/model.joblib') != expected_model_hash:
        return False
    if file_sha256(pointer) == expected_pointer_hash:
        return True
    try:
        active=json.loads(pointer.read_text(encoding='utf-8'))
        directory=active['directory']
        for _ in range(12):
            if not re.fullmatch(r'v\d+(?:\.\d+)*',directory):
                return False
            version=root/'artifacts/versions'/directory
            manifest=json.loads((version/'manifest.json').read_text(encoding='utf-8'))
            if not manifest.get('ui_only') or manifest['model_sha256'] != expected_model_hash:
                return False
            if file_sha256(version/'model.joblib') != expected_model_hash:
                return False
            parent=root/'artifacts/versions'/manifest['parent']
            previous=json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
            science=manifest.get('scientific_source_hashes',{})
            if not science or any(previous['source_files'].get(name)!=digest or file_sha256(root/name)!=digest for name,digest in science.items()):
                return False
            if file_sha256(version/'parent_active_version.json') == expected_pointer_hash:
                return True
            directory=manifest['parent']
    except (OSError,KeyError,ValueError):
        return False
    return False
