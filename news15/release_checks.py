"""Allow an audited UI successor while retaining the same scientific release."""
import json
import re
from pathlib import Path

from .data import file_sha256
from .paths import PROJECT_DIR


def verified_retraining(expected_baseline_hash, project_dir=PROJECT_DIR):
    """Check deliberate retraining without claiming the active model stayed identical."""
    root=Path(project_dir)
    try:
        current=json.loads((root/'artifacts/active_version.json').read_text(encoding='utf-8'))
        if current.get('model_file')!='artifacts/model.joblib' or current.get('report_file')!='reports/training_report.json':
            return False
        if current.get('corpus_import_model_sha256',current.get('baseline_model_sha256'))!=expected_baseline_hash:
            return False
        if file_sha256(root/current['model_file'])!=current.get('model_sha256'):
            return False
        if file_sha256(root/current['report_file'])!=current.get('report_sha256'):
            return False
        report=json.loads((root/current['report_file']).read_text(encoding='utf-8'))
        return (report['metadata'].get('corpus_import_model_sha256',report['metadata']['baseline_model_sha256'])==expected_baseline_hash
            and report['freeze']['fingerprints']==current['fingerprints']
            and report['checks']['frozen_before_external_labels'] is True
            and report['checks']['group_roles_disjoint'] is True)
    except (OSError,KeyError,ValueError):
        return False


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
