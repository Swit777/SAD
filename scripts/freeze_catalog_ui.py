"""Freeze UI 2.0.2 with the existing scientific model and pinned catalog files."""
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from news15.paths import PROJECT_DIR,ARTIFACTS_DIR
from news15.data import file_sha256
from news15.corpus_expansion import write_json

parent=ARTIFACTS_DIR/'versions/v2.0.1'
target=ARTIFACTS_DIR/'versions/v2.0.2'
if target.exists():raise SystemExit('UI release already exists; it will not be overwritten.')
manifest=json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
if file_sha256(ARTIFACTS_DIR/'model.joblib')!=manifest['model_sha256']:
    raise RuntimeError('The scientific model changed.')
science={key:digest for key,digest in manifest['source_files'].items() if key.replace('\\','/') in {
    'news15/data.py','news15/model.py','news15/consensus.py','news15/audit.py','news15/streaming.py','protocols/v2.json'}}
if any(file_sha256(PROJECT_DIR/name)!=digest for name,digest in science.items()):
    raise RuntimeError('Scientific source changed in a UI-only release.')
target.mkdir(parents=True)
shutil.copy2(ARTIFACTS_DIR/'active_version.json',target/'parent_active_version.json')
shutil.copy2(parent/'model.joblib',target/'model.joblib')
shutil.copytree(parent/'reports',target/'reports')
files=sorted((PROJECT_DIR/'news15').glob('*.py'))+sorted((PROJECT_DIR/'scripts').glob('*.py'))+sorted((PROJECT_DIR/'tests').glob('*.py'))
files += [PROJECT_DIR/name for name in ['app.py','README.md','requirements.txt','protocols/v2.json']]
sources={}
for file in files:
    relative=file.relative_to(PROJECT_DIR)
    dest=target/'source'/relative
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(file,dest)
    sources[str(relative)]=file_sha256(file)
manifest.update({'version':'2.0.2','model_version':'2.0.0','parent':'v2.0.1','ui_only':True,
    'amendment':'Expanded catalog v4, bounded metadata search and record preview, report and presentation downloads. Scientific model and audit results preserved.',
    'source_files':sources,'scientific_source_hashes':science,
    'parent_pointer_sha256':file_sha256(target/'parent_active_version.json'),
    'data_files':{name:file_sha256(PROJECT_DIR/name) for name in ['data/processed/corpus_v4/articles.parquet','data/processed/corpus_v4/claims.parquet','data/processed/corpus_v4/manifest.json','reports/multisource_expansion/import_report.json']},
    'deliverables':{name:file_sha256(PROJECT_DIR/name) for name in ['deliverables/project_report.docx','deliverables/project_presentation.pptx']},
    'finalized':True})
write_json(target/'manifest.json',manifest)
write_json(ARTIFACTS_DIR/'active_version.json',{'version':'2.0.2','directory':'v2.0.2','model_version':'2.0.0'})
print('Frozen UI 2.0.2, catalog v4 and final documents. Scientific model 2.0.0 preserved.')
