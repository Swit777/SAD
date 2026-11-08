"""Exercise every Streamlit page and the new catalog workflow."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
from news15.paths import PROJECT_DIR

app = AppTest.from_file(str(PROJECT_DIR/'app.py'),default_timeout=90).run()
results = {}
sections = list(app.radio(key='section_navigation').options)
for section in sections:
    app.radio(key='section_navigation').set_value(section).run()
    results[section] = {"exceptions":[str(e.value) for e in app.exception],"metrics":[{"label":m.label,"value":m.value} for m in app.metric]}
app.radio(key='section_navigation').set_value('Каталог данных').run()
app.selectbox(key='catalog_source_articles').set_value('FakeNewsCorpus').run()
app.button(key='FormSubmitter:catalog_filters_articles-Найти записи').click().run()
app.button(key='open_record_articles').click().run()
results['catalog_open'] = {"exceptions":[str(e.value) for e in app.exception],"text_loaded":bool(app.session_state['catalog_open_record']['record']['text']),"scope":app.session_state['catalog_open_record']['record']['label_scope']}
opened = dict(app.session_state['catalog_open_record']['record'])
next(button for button in app.button if button.label == 'Открыть для анализа моделью').click().run()
results['catalog_to_analysis'] = {
    "exceptions":[str(e.value) for e in app.exception],
    "navigation":app.radio(key='section_navigation').value,
    "same_text":app.text_area(key='article_text').value == opened['text'],
    "same_title":app.text_input(key='article_title').value == opened['title'],
}
assert results['catalog_to_analysis']['same_text'] and results['catalog_to_analysis']['same_title']
app.radio(key='section_navigation').set_value('Материалы проекта').run()
downloads = [element.label for element in app.get('download_button')]
results['materials'] = {"exceptions":[str(e.value) for e in app.exception],"download_labels":downloads}
assert 'Скачать отчёт Word' in downloads and 'Скачать презентацию PowerPoint' in downloads
path = PROJECT_DIR/'reports/.browser-qa'
path.mkdir(parents=True,exist_ok=True)
(path/'ui_checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
failed=[key for key,value in results.items() if value['exceptions']]
print(json.dumps({'pages':len(sections),'failed':failed,'catalog_scope':results['catalog_open']['scope']},ensure_ascii=True))
if failed:raise SystemExit('UI verification failed')
