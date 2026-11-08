"""Local Streamlit interface for topic 15; no network access on page reruns."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from news15 import data, experiments, model, streaming
from news15.paths import MODEL_PATH, PROCESSED_DIR, REPORTS_DIR


st.set_page_config(
    page_title="News 15 · Анализ новостей",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1260px; padding-top: 4.5rem; padding-bottom: 3rem;}
      h1 {font-size: 2.35rem !important; letter-spacing: -.065rem;}
      h2 {font-size: 1.45rem !important; letter-spacing: -.025rem;}
      h3 {font-size: 1.08rem !important;}
      [data-testid="stMetricValue"] {font-size: 1.85rem;}
      [data-testid="stSidebar"] {border-right: 1px solid #e1e6df;}
      .eyebrow {color: #39735c; font-size: .76rem; letter-spacing: .15rem;
                 font-weight: 700; margin-bottom: .3rem;}
      .lead {font-size: 1.06rem; color: #56625a; max-width: 820px; line-height: 1.65;}
      .note {border-left: 3px solid #56836c; padding: .65rem 1rem;
              background: #edf2ec; color: #334c3e; border-radius: 0 6px 6px 0;}
      [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px;}
      div[data-testid="stForm"] {border-color: #dce3da;}
      button[kind="primary"] {border-radius: 7px;}
    </style>
    """,
    unsafe_allow_html=True,
)

SECTIONS = [
    "Обзор",
    "Проверка новости",
    "Модели и данные",
    "Эксперименты",
    "Поток и обратная связь",
    "Методологический аудит",
]
DECISIONS = {"real": "Вероятно истинная", "fake": "Вероятно фейковая", "review": "Нужна проверка"}
LABELS = {0: "Истинные", 1: "Фейковые"}


def version_of(path: Path) -> tuple:
    path = Path(path)
    if not path.exists():
        return ()
    if path.is_file():
        info = path.stat()
        return (info.st_mtime_ns, info.st_size)
    return tuple(
        (item.name, item.stat().st_mtime_ns, item.stat().st_size)
        for item in sorted(path.iterdir())
        if item.is_file()
    )


@st.cache_data(show_spinner=False)
def cached_articles(version: tuple) -> pd.DataFrame:
    return data.load_articles()


@st.cache_resource(show_spinner=False)
def cached_bundle(version: tuple) -> dict:
    return model.load_bundle()


@st.cache_data(show_spinner=False)
def cached_summary(version: tuple) -> dict:
    return data.dataset_summary()


def resources() -> tuple[pd.DataFrame, dict | None, dict, list[str]]:
    articles, bundle, summary, errors = pd.DataFrame(), None, {}, []
    processed_version = version_of(Path(PROCESSED_DIR))
    if processed_version:
        try:
            articles = cached_articles(processed_version)
            summary = cached_summary(processed_version)
        except Exception as exc:
            errors.append(f"Данные: {exc}")
    if Path(MODEL_PATH).is_file():
        try:
            bundle = cached_bundle(version_of(Path(MODEL_PATH)))
        except Exception as exc:
            errors.append(f"Модель: {exc}")
    return articles, bundle, summary, errors


def comparison_frame(bundle: dict | None) -> pd.DataFrame:
    if not bundle:
        return pd.DataFrame()
    value = bundle.get("comparison")
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value or [])


def selected_record(bundle: dict | None) -> dict:
    frame = comparison_frame(bundle)
    if frame.empty:
        return {}
    metadata = bundle.get("metadata", {})
    for key in ("selected_model", "model_name", "best_model", "selected_model_name"):
        name = metadata.get(key)
        if name and "model" in frame and name in frame["model"].values:
            return frame.loc[frame["model"] == name].iloc[0].to_dict()
    if "validation_f1_macro" in frame:
        frame = frame.sort_values("validation_f1_macro", ascending=False)
    return frame.iloc[0].to_dict()


def percent(value: object) -> str:
    try:
        return f"{float(value):.1%}" if pd.notna(value) else "—"
    except (TypeError, ValueError):
        return "—"


def integer(value: object) -> str:
    try:
        return f"{int(value):,}".replace(",", " ")
    except (TypeError, ValueError):
        return "—"


def csv_download(frame: pd.DataFrame, name: str, label: str = "Скачать CSV") -> None:
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=name,
        mime="text/csv",
        key=f"download_{name}",
    )


def require_model(bundle: dict | None) -> bool:
    if bundle is not None:
        return True
    st.info("Сначала подготовьте данные и обучите модели в разделе «Модели и данные».")
    return False


def overview(articles: pd.DataFrame, bundle: dict | None, summary: dict) -> None:
    st.markdown('<div class="eyebrow">ТЕМА 15 · FAKE NEWS DETECTION</div>', unsafe_allow_html=True)
    st.title("Что видит модель в новостях")
    st.markdown(
        '<p class="lead">Исследование классификации англоязычных новостей: '
        'от реального корпуса и сравнения моделей до объяснений, '
        'проверки устойчивости и обратной связи.</p>',
        unsafe_allow_html=True,
    )
    record = selected_record(bundle)
    audited = bool(bundle and bundle.get("schema_version") == 2)
    deployed_metrics = bundle.get("metadata", {}).get("external_metrics" if audited else "test_metrics", {}) if bundle else {}
    cols = st.columns(4)
    cols[0].metric("Новостей после очистки", integer(len(articles)) if len(articles) else "—")
    cols[1].metric("Внешний macro-F1" if audited else "Макро-F1 на ISOT", percent(deployed_metrics.get("f1_macro", record.get("test_f1_macro"))))
    cols[2].metric("Внешняя accuracy" if audited else "Точность на ISOT", percent(deployed_metrics.get("accuracy", record.get("test_accuracy"))))
    cols[3].metric("Участников консенсуса" if audited else "Сравнено моделей", "3" if audited else integer(len(comparison_frame(bundle))) if bundle else "—")
    if audited:
        st.warning("Версия 2 — исследовательская. Главные оценки относятся к независимым raw заголовкам FakeNewsNet. Автоматические пользовательские вердикты отключены.")
        st.caption("Полные тексты ISOT и внешние заголовки — разные единицы проверки. Сопоставление заголовок ↔ заголовок и доверительные интервалы приведены в методологическом аудите.")
    if bundle:
        st.caption(
            f"Модель: {bundle['metadata'].get('selected_model', 'обученный классификатор')}. "
            + ("Настройка, калибровка и порог используют разные подвыборки; старый ISOT test — повторная диагностика." if audited else "Выбор модели и порога использовал одну validation; внешняя валидация в v1 отсутствовала.")
        )
    else:
        st.info("Результаты появятся после подготовки корпуса и обучения. Эти действия запускаются вручную.")
    st.markdown(
        '<div class="note">Модель оценивает текст по закономерностям корпуса ISOT. '
        'Её оценка не подтверждает факты. Ввод и модель — на английском языке.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Путь новости")
        with st.container(border=True):
            st.markdown("**01 · Текст**  →  заголовок и содержание новости")
            st.markdown("**02 · Оценка**  →  вероятность класса и зона ручной проверки")
            st.markdown("**03 · Объяснение**  →  слова, повлиявшие на решение")
            st.markdown("**04 · Обратная связь**  →  изменение порога по проверенным меткам")
    with right:
        st.subheader("Состав корпуса")
        if articles.empty:
            st.caption("Пока нет подготовленного датасета.")
        else:
            counts = articles.groupby(["split", "label"]).size().unstack(fill_value=0)
            counts = counts.reindex([x for x in ["train", "validation", "test"] if x in counts.index])
            counts = counts.rename(columns=LABELS)
            st.bar_chart(counts, color=["#3e7860", "#cb8558"][: len(counts.columns)])
            st.caption("Группы дубликатов не должны пересекать выборки. Подробности очистки — в «Модели и данные».")
    if bundle and isinstance(bundle.get("test_predictions"), pd.DataFrame):
        predictions = bundle["test_predictions"]
        if not predictions.empty and {"label", "probability_fake"}.issubset(predictions.columns):
            st.subheader("Какие ошибки остаются")
            threshold = float(bundle.get("threshold", 0.5))
            predicted = (predictions["probability_fake"] >= threshold).astype(int)
            confusion = pd.crosstab(predictions["label"], predicted).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            confusion.index = ["Факт: истинная", "Факт: фейковая"]
            confusion.columns = ["Прогноз: истинная", "Прогноз: фейковая"]
            a, b = st.columns([1.1, 1])
            a.dataframe(confusion, width="stretch")
            b.caption(
                f"Матрица ошибок на отложенном тесте при исходном пороге {threshold:.2f}. "
                "Изменение порога в интерактивной проверке не меняет опубликованные результаты."
            )


def choose_example(articles: pd.DataFrame, label: int) -> None:
    candidates = articles[(articles["split"] == "test") & (articles["label"] == label)]
    if candidates.empty:
        return
    offset_key = f"example_offset_{label}"
    offset = int(st.session_state.get(offset_key, 0))
    row = candidates.iloc[offset % len(candidates)]
    st.session_state["article_title"] = str(row["title"])
    st.session_state["article_text"] = str(row["text"])
    st.session_state[offset_key] = offset + 1
    st.session_state.pop("article_result", None)


def check_article(articles: pd.DataFrame, bundle: dict | None) -> None:
    st.title("Проверка новости")
    st.caption("Вставьте английский текст или возьмите настоящий пример из отложенной выборки.")
    if not require_model(bundle):
        return
    if not articles.empty:
        first, second, _ = st.columns([1, 1, 1.5])
        first.button("Пример: истинная", on_click=choose_example, args=(articles, 0), width="stretch")
        second.button("Пример: фейковая", on_click=choose_example, args=(articles, 1), width="stretch")
        st.caption("Метки примеров взяты из ISOT; примеры не участвовали в обучении.")
    with st.expander("Порог решения"):
        a, b = st.columns(2)
        audited = bundle.get("schema_version") == 2
        threshold = a.slider("Порог класса «фейк»", 0.10, 0.90, float(bundle.get("threshold", 0.5)), 0.01, disabled=audited)
        review_margin = b.slider("Зона проверки вокруг порога", 0.00, 0.25, float(bundle.get("consensus_policy", {}).get("margin", .10)), 0.01, disabled=audited)
        st.caption("В версии 2 порог и минимальное согласие зафиксированы до внешней проверки." if audited else "Внутри зоны неопределённости приложение предлагает ручную проверку.")
    with st.form("article_form"):
        title = st.text_input("Заголовок", key="article_title", max_chars=2000, placeholder="English headline")
        text = st.text_area("Текст новости", key="article_text", max_chars=120000, height=235, placeholder="Paste the English article here…")
        submitted = st.form_submit_button("Проанализировать", type="primary", width="stretch")
    if submitted:
        if not (title.strip() or text.strip()):
            st.warning("Добавьте заголовок или текст новости.")
        else:
            try:
                with st.spinner("Считаем оценку и вклад слов…"):
                    result = model.predict_article(bundle, title, text, threshold=threshold, review_margin=review_margin)
                    explanation = model.explain_article(bundle, title, text, top_n=12)
                st.session_state["article_result"] = {
                    "result": result, "explanation": explanation,
                    "title": title, "text": text, "threshold": threshold, "margin": review_margin,
                }
            except Exception as exc:
                st.error(f"Не удалось обработать новость: {exc}")
    stored = st.session_state.get("article_result")
    if not stored:
        return
    if (stored["title"], stored["text"], stored["threshold"], stored["margin"]) != (title, text, threshold, review_margin):
        st.caption("Текст или порог изменён. Нажмите «Проанализировать», чтобы обновить результат.")
        return
    result = stored["result"]
    if result.get("review_reason"):
        st.info(result["review_reason"])
    decision = result.get("decision", "review")
    st.subheader(DECISIONS.get(decision, "Результат анализа"))
    score = float(result["probability_fake"])
    a, b = st.columns([1, 2])
    a.metric("Оценка класса «фейк»", percent(score))
    b.progress(min(1.0, max(0.0, score)), text="Оценка в сторону класса «фейк»")
    st.caption("Это вероятность класса по обученной модели; она не заменяет проверку источников и фактов.")
    for warning in result.get("warnings", []):
        st.warning(str(warning))
    if result.get("members"):
        st.dataframe(pd.DataFrame(result["members"]), hide_index=True, width="stretch")
        st.caption(f"Согласие участников: {percent(result['agreement'])}. Участники обучены на одном корпусе и не являются независимыми проверками фактов.")
    explanation = stored["explanation"]
    if not explanation.empty:
        st.subheader("Почему получилась такая оценка")
        st.caption("В v2 показан точный interventional SHAP логита первого откалиброванного участника. Он не объясняет весь нелинейный консенсус и не доказывает факты." if bundle.get("schema_version") == 2 else "Положительный вклад сдвигает решение к классу «фейк», отрицательный — к классу «истинная».")
        chart = explanation[["feature", "contribution"]].copy().sort_values("contribution")
        st.bar_chart(chart.set_index("feature"), horizontal=True, color="#4c7b65", height=360)
        with st.expander("Вклады признаков в таблице"):
            st.dataframe(explanation, hide_index=True, width="stretch")
            csv_download(explanation, "article_explanation.csv")


def models_and_data(articles: pd.DataFrame, bundle: dict | None, summary: dict, errors: list[str]) -> None:
    st.title("Модели и данные")
    st.caption("Корпус, очистка, разделение выборок и воспроизводимое сравнение классификаторов.")
    st.subheader("Реальный корпус ISOT")
    st.markdown(
        "Англоязычные новости с исходными метками **real / fake**. "
        f"Источник: [ISOT Research Lab, University of Victoria]({data.ISOT_DESCRIPTION}). "
        "Оценка на этом корпусе показывает качество на его новостях; перенос на новые СМИ, языки и периоды требует отдельной проверки."
    )
    if summary:
        a, b, c, d = st.columns(4)
        a.metric("Исходных записей", integer(summary.get("raw_rows")))
        b.metric("После очистки", integer(summary.get("clean_rows", len(articles))))
        c.metric("Удалено дубликатов", integer(summary.get("removed_duplicates")))
        d.metric("Спорных групп", integer(summary.get("ambiguous_groups")))
        if "date_range" in summary:
            dates = summary["date_range"]
            if isinstance(dates, dict):
                dates = " — ".join(str(value)[:10] for value in dates.values())
            elif isinstance(dates, (list, tuple)):
                dates = " — ".join(str(value)[:10] for value in dates)
            st.caption(f"Даты публикаций: {dates}. Без распознанной даты: {summary.get('missing_dates', 0)}.")
    with st.expander("Подготовка и обучение", expanded=articles.empty or bundle is None):
        st.caption("Загрузка и обучение запускаются только по кнопке. Подготовка полного корпуса требует интернета один раз.")
        first, second = st.columns(2)
        if first.button("Загрузить и подготовить ISOT", width="stretch"):
            try:
                with st.spinner("Загружаем архив и готовим выборки…"):
                    data.download_isot()
                    data.prepare_isot()
                cached_articles.clear()
                cached_summary.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Подготовка не завершена: {exc}")
        if second.button("Обучить и сравнить модели", disabled=articles.empty or bool(bundle and bundle.get("schema_version") == 2), type="primary", width="stretch"):
            try:
                with st.spinner("Обучаем модели и оцениваем отложенные выборки…"):
                    model.train_models()
                cached_bundle.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Обучение не завершено: {exc}")
        if bundle and bundle.get("schema_version") == 2:
            st.code("python scripts/audit_cycle.py", language="powershell")
            st.caption("Версия 2 пересчитывается по фиксированному протоколу. Повторное использование внешнего набора будет отмечено в отчёте.")
    for error in errors:
        st.error(error)
    if not articles.empty:
        st.subheader("Разделение данных")
        split_table = articles.groupby("split").agg(Новостей=("id", "size"), Групп=("group_id", "nunique"))
        split_table = split_table.reindex([x for x in ["train", "validation", "test"] if x in split_table.index])
        st.dataframe(split_table, width="stretch")
        with st.expander("Посмотреть новости"):
            selected_split = st.selectbox("Выборка", ["test", "validation", "train"], key="preview_split")
            columns = [x for x in ["title", "label", "subject", "published_at"] if x in articles]
            preview = articles.loc[articles["split"] == selected_split, columns].head(30).copy()
            preview["label"] = preview["label"].map({0: "Истинная", 1: "Фейковая"})
            st.dataframe(preview, hide_index=True, width="stretch")
    comparison = comparison_frame(bundle)
    if not comparison.empty:
        st.subheader("Сравнение моделей")
        st.caption("Основная метрика — макро-F1. Сравнение ниже рассчитано при пороге 0.50. Для выбранной модели отдельный порог подбирается на validation; итог с этим порогом показан в «Обзор».")
        names = {
            "model": "Модель", "validation_f1_macro": "Validation · макро-F1",
            "test_f1_macro": "Test · макро-F1", "test_accuracy": "Test · accuracy",
            "test_precision": "Test · precision", "test_recall": "Test · recall", "test_roc_auc": "Test · ROC AUC",
        }
        shown = comparison[[key for key in names if key in comparison]].rename(columns=names)
        st.dataframe(shown, hide_index=True, width="stretch")
        available = [key for key in ["validation_f1_macro", "test_f1_macro"] if key in comparison]
        if available and "model" in comparison:
            st.bar_chart(comparison.set_index("model")[available].rename(columns=names), horizontal=True, color=["#9eb8a6", "#3e7860"][:len(available)])
        csv_download(comparison, "model_comparison.csv", "Скачать сравнение моделей")
        with st.expander("Метаданные обучения"):
            st.json(bundle.get("metadata", {}), expanded=False)


def read_experiments() -> dict:
    path = Path(REPORTS_DIR) / "experiments.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def report_frame(value: object) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
        return pd.DataFrame(value)
    if isinstance(value, dict) and value and all(isinstance(row, dict) for row in value.values()):
        return pd.DataFrame.from_dict(value, orient="index").rename_axis("Вариант").reset_index()
    return pd.DataFrame()


REPORT_COLUMNS = {
    "condition": "Условие", "method": "Метод", "subject": "Тематика", "length_bin": "Длина текста", "group": "Группа",
    "n": "Новостей", "real_count": "Истинных", "fake_count": "Фейковых",
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "precision_fake": "Precision · фейк",
    "recall_fake": "Recall · фейк", "f1_fake": "F1 · фейк", "f1_macro": "Макро-F1", "roc_auc": "ROC AUC",
    "false_positive_rate": "Ложный фейк · FPR", "false_negative_rate": "Пропущенный фейк · FNR",
    "average_precision": "Average precision", "brier_score": "Brier score",
    "delta_f1_vs_original": "Изменение F1", "prediction_flip_rate": "Доля изменённых решений",
    "mean_absolute_probability_change": "Среднее изменение оценки", "training_n": "Обучающих примеров",
    "fit_n": "Векторов в обучении", "fit_real_count": "Векторов · real", "fit_fake_count": "Векторов · fake",
    "split": "Выборка", "label": "Класс", "reuters_marker_rate": "Доля с маркером Reuters",
}
REPORT_VALUES = {
    "original": "Исходный текст", "uppercase": "Верхний регистр",
    "word_dropout_20pct": "Удалено 20% слов", "character_typos_8pct_words": "Опечатки в 8% слов",
    "raw_text": "Исходный текст", "source_cleaned_text": "Маркеры источника удалены",
    "imbalanced_unweighted": "Дисбаланс · без коррекции", "imbalanced_class_weight": "Веса классов",
    "imbalanced_smote": "SMOTE на train",
}


def show_report_value(value: object, key: str, depth: int = 0) -> None:
    frame = report_frame(value)
    if not frame.empty:
        display = frame.copy()
        for column in display:
            if display[column].map(lambda item: isinstance(item, (dict, list))).any():
                display[column] = display[column].map(lambda item: json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else item)
        display = display.replace(REPORT_VALUES).rename(columns=REPORT_COLUMNS)
        st.dataframe(display, hide_index=True, width="stretch")
        numeric = [c for c in ["f1_macro", "accuracy", "f1_fake"] if c in frame.select_dtypes(include="number").columns]
        categories = [c for c in frame.columns if c not in frame.select_dtypes(include="number").columns and not frame[c].map(lambda item: isinstance(item, (dict, list))).any()]
        if numeric and categories:
            chart = frame.set_index(categories[0])[numeric[:3]]
            chart.index = chart.index.map(lambda item: REPORT_VALUES.get(item, item))
            chart = chart.rename(columns=REPORT_COLUMNS)
            st.bar_chart(chart, horizontal=True)
        return
    if isinstance(value, dict) and depth < 2:
        scalars = {k: v for k, v in value.items() if not isinstance(v, (dict, list))}
        if scalars:
            st.dataframe(pd.DataFrame({"Показатель": [REPORT_COLUMNS.get(k, k) for k in scalars], "Значение": [str(v) for v in scalars.values()]}), hide_index=True, width="stretch")
        for child, nested in value.items():
            if isinstance(nested, (dict, list)):
                st.markdown(f"**{str(child).replace('_', ' ')}**")
                show_report_value(nested, f"{key}_{child}", depth + 1)
    elif value is not None:
        st.json(value, expanded=False) if isinstance(value, (dict, list)) else st.write(value)


def show_decision_boundary(report: dict) -> None:
    files = report.get("files", {})
    if not files.get("points") or not files.get("surface"):
        return
    points_path, surface_path = Path(files["points"]), Path(files["surface"])
    if not points_path.is_file() or not surface_path.is_file():
        st.caption("Для графика нужны сохранённые CSV эксперимента. Запустите расчёт повторно.")
        return
    points, surface = pd.read_csv(points_path), pd.read_csv(surface_path)
    grid = surface.pivot(index="component_2", columns="component_1", values="probability_fake")
    figure = go.Figure(go.Heatmap(
        x=grid.columns, y=grid.index, z=grid.to_numpy(), zmin=0, zmax=1,
        colorscale=[[0, "#c4ddd1"], [0.5, "#f7f6eb"], [1, "#e8bf9e"]],
        colorbar={"title": "P(fake)", "thickness": 13},
        hovertemplate="Компонента 1: %{x:.2f}<br>Компонента 2: %{y:.2f}<br>P(fake): %{z:.2f}<extra></extra>",
    ))
    threshold = float(report.get("threshold", 0.5))
    figure.add_trace(go.Contour(
        x=grid.columns, y=grid.index, z=grid.to_numpy(), showscale=False,
        contours={"start": threshold, "end": threshold, "size": 1, "coloring": "none"},
        line={"color": "#34483d", "width": 2}, hoverinfo="skip",
    ))
    test_points = points.loc[points["split"] == "test"]
    for label, name, color in [(0, "Истинные · test", "#357657"), (1, "Фейковые · test", "#b66a36")]:
        selected = test_points.loc[test_points["label"] == label]
        figure.add_trace(go.Scattergl(
            x=selected["component_1"], y=selected["component_2"], mode="markers", name=name,
            marker={"size": 5, "color": color, "opacity": 0.65, "line": {"width": 0.4, "color": "white"}},
            hovertemplate="Компонента 1: %{x:.2f}<br>Компонента 2: %{y:.2f}<extra>%{fullData.name}</extra>",
        ))
    figure.update_layout(
        height=450, margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_title="SVD-компонента 1 · стандартизована", yaxis_title="SVD-компонента 2 · стандартизована",
        legend={"orientation": "h", "y": -0.2}, paper_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(range=[float(grid.columns.min()), float(grid.columns.max())])
    figure.update_yaxes(range=[float(grid.index.min()), float(grid.index.max())])
    st.plotly_chart(figure, width="stretch")


def experiment_page(articles: pd.DataFrame, bundle: dict | None) -> None:
    st.title("Эксперименты")
    st.caption("Измеряем устойчивость, смещения корпуса и поведение модели рядом с границей решения.")
    st.markdown(
        "Каждый эксперимент использует фиксированные выборки. Изменение качества сравнивается с исходной моделью; "
        "искусственные искажения отмечаются как экспериментальные."
    )
    if st.button("Выполнить эксперименты", type="primary", disabled=bundle is None or articles.empty):
        try:
            with st.spinner("Проводим эксперименты…"):
                st.session_state["experiment_report"] = experiments.run_experiments()
        except Exception as exc:
            st.error(f"Эксперименты не завершены: {exc}")
    report = st.session_state.get("experiment_report") or read_experiments()
    if bundle and bundle.get("schema_version") == 2 and report.get("metadata", {}).get("model_version") != "2.0.0":
        st.info("Ниже архивные эксперименты v1. Для версии 2 основные проверки находятся в «Методологический аудит»; этот набор можно пересчитать кнопкой.")
    if not report:
        st.info("Сохранённых результатов пока нет. Подготовьте модель и запустите эксперименты.")
        return
    titles = {
        "bias": "Качество по группам", "noise": "Устойчивость к шуму",
        "source_marker_ablation": "Влияние маркеров источника",
        "smote": "Контролируемый дисбаланс и SMOTE",
        "decision_boundary": "Граница решения учебной 2D-модели",
        "baseline_metrics": "Исходная модель · экспериментальная подвыборка",
        "metadata": "Параметры экспериментов",
    }
    notes = {
        "bias": "Тематика и длина текста могут быть обходными признаками. Эти срезы корпуса не доказывают политическую или демографическую справедливость. В группах с одним классом ROC AUC не определён.",
        "noise": "Искажения синтетические и применены к одним и тем же тестовым текстам. Они проверяют устойчивость только к этим изменениям. Нормализация регистра может сделать uppercase нейтральным.",
        "source_marker_ablation": "Сравниваются отдельно обученные на одинаковой подвыборке модели. Удаление Reuters и других явных маркеров не устраняет все различия стиля и тематики между источниками.",
        "smote": "В исходном ISOT классы почти сбалансированы. Здесь искусственно создано около 20% фейков в train. SMOTE добавляет числовые SVD-векторы; это не новые реальные новости. Тест не меняется, улучшение не предполагается заранее.",
        "decision_boundary": "Это реальная граница отдельной логистической модели на двух SVD-компонентах, обученных только на train. Она служит учебной визуализацией и не является границей основного TF-IDF-классификатора. На графике показаны тестовые точки; крайние координаты обрезаны.",
        "baseline_metrics": "Для воспроизводимых экспериментов используется ограниченная подвыборка. Эти метрики могут отличаться от результатов на полном тесте в разделе «Обзор».",
    }
    subtitles = {"by_subject": "По тематике", "by_length": "По длине текста", "results": "Сравнение условий", "source_marker_rates": "Частота маркера Reuters", "metrics": "Метрики учебной модели"}
    for key, value in report.items():
        if key in ["report_file", "created_at", "generated_at", "schema_version"]:
            continue
        if key == "streaming_rtap":
            st.caption("Поток и RTAP доступны в отдельном разделе «Поток и обратная связь».")
            continue
        with st.expander(titles.get(key, str(key).replace("_", " ").capitalize()), expanded=key != "metadata"):
            if key in notes:
                st.caption(notes[key])
            if key == "decision_boundary":
                show_decision_boundary(value)
            if isinstance(value, dict) and key not in ["baseline_metrics", "metadata"]:
                for child, nested in value.items():
                    if child in subtitles:
                        st.markdown(f"**{subtitles[child]}**")
                        show_report_value(nested, f"{key}_{child}")
                if key == "smote":
                    st.caption(f"Синтетических векторов: {integer(value.get('synthetic_vector_count'))}. Доля fake в обучении: {percent(value.get('controlled_training_fake_fraction'))}.")
            else:
                show_report_value(value, key)
    st.download_button(
        "Скачать отчёт экспериментов", json.dumps(report, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        file_name="experiments.json", mime="application/json",
    )


def reset_stream(bundle: dict) -> None:
    st.session_state["stream_policy"] = streaming.AdaptiveReviewPolicy(
        initial_threshold=float(bundle.get("threshold", 0.5)), min_threshold=0.2, max_threshold=0.8,
    )
    st.session_state["stream_events"] = []
    st.session_state["stream_cursor"] = 0
    st.session_state["stream_feedback_notice"] = None
    st.session_state["stream_session_id"] = uuid4().hex[:12]
    st.session_state.pop("stream_selected_event", None)


def stream_feedback(event_index: int, label: int, source: str) -> None:
    events = st.session_state["stream_events"]
    event = events[event_index]
    if event.get("feedback_label") is not None:
        return
    policy = st.session_state["stream_policy"]
    update = policy.observe(event["probability_fake"], int(label), event_id=event["event_id"])
    event["feedback_label"] = int(label)
    event["feedback_source"] = source
    st.session_state["stream_feedback_notice"] = update
    streaming.append_event({"type": "feedback", "event_id": event["event_id"], "label": int(label), "source": source, "policy_update": update})


def saved_stream_experiment() -> None:
    path = Path(REPORTS_DIR) / "stream_replay.json"
    if not path.is_file():
        return
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    metrics = report.get("metrics", {})
    if not metrics:
        return
    with st.expander("Архивный эксперимент · фиксированный и адаптивный порог"):
        st.caption("Симуляция обратной связи метками ISOT: сначала записывается прогноз, затем становится доступна исходная метка. Это учебный эксперимент на архиве; он не заменяет ручную проверку новых новостей.")
        rows = []
        metric_names = {
            "accuracy": "Accuracy", "f1_macro": "Макро-F1", "precision_fake": "Precision · фейк",
            "recall_fake": "Recall · фейк", "review_fraction": "Доля на проверку",
            "automatic_fraction": "Доля автоматических решений", "automatic_accuracy": "Accuracy автоматических решений",
        }
        for key, name in [("fixed", "Фиксированный порог"), ("adaptive", "Адаптивный порог")]:
            if key in metrics:
                rows.append({"Режим": name, **{label: metrics[key].get(metric) for metric, label in metric_names.items()}})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        changes = report.get("threshold_changes", [])
        changes_count = len(changes) if isinstance(changes, list) else changes
        st.caption(
            f"Новостей: {report.get('events', '—')}. Изменений порога: {changes_count}. "
            f"Начальный порог: {report.get('initial_threshold', 0.5):.2f}; конечный: {report.get('final_threshold', 0.5):.2f}."
        )


def stream_page(articles: pd.DataFrame, bundle: dict | None) -> None:
    st.title("Поток и обратная связь")
    st.caption("Воспроизведение отложенных новостей из ISOT. Каждое нажатие обрабатывает следующую порцию.")
    if not require_model(bundle) or articles.empty:
        return
    saved_stream_experiment()
    if "stream_policy" not in st.session_state:
        reset_stream(bundle)
    policy = st.session_state["stream_policy"]
    test = articles[articles["split"] == "test"].sort_values("published_at", na_position="last")
    if test.empty:
        st.info("В датасете нет тестовой выборки для воспроизведения.")
        return
    with st.expander("Настройки воспроизведения", expanded="stream_queue" not in st.session_state):
        a, b = st.columns(2)
        count = a.number_input("Новостей в очереди", 1, min(1000, len(test)), min(100, len(test)), step=1)
        batch = b.number_input("Новостей за один шаг", 1, 30, 5, step=1)
        if st.button("Создать новую очередь"):
            reset_stream(bundle)
            st.session_state["stream_queue"] = test.head(int(count)).to_dict("records")
            st.rerun()
        st.caption("Новая очередь сбрасывает текущую сессию обратной связи. Обученная модель и итоговые тестовые метрики сохраняются.")
    queue = st.session_state.get("stream_queue", [])
    if not queue:
        st.info("Создайте очередь, чтобы начать воспроизведение.")
        return
    cursor = st.session_state["stream_cursor"]
    events = st.session_state["stream_events"]
    a, b, c, d = st.columns(4)
    a.metric("Обработано", f"{cursor} / {len(queue)}")
    b.metric("На ручную проверку", sum(event["decision"] == "review" for event in events))
    c.metric("Подтверждённых меток", sum(event.get("feedback_label") is not None for event in events))
    d.metric("Текущий порог", f"{policy.threshold:.2f}")
    if st.button(f"Обработать следующие {min(int(batch), len(queue) - cursor)}", type="primary", disabled=cursor >= len(queue)):
        try:
            with st.spinner("Обрабатываем новости…"):
                for index in range(cursor, min(cursor + int(batch), len(queue))):
                    row = queue[index]
                    started = perf_counter()
                    result = model.predict_article(bundle, str(row["title"]), str(row["text"]), threshold=policy.threshold)
                    latency_ms = (perf_counter() - started) * 1000
                    probability = float(result["probability_fake"])
                    event = {
                        "event_id": f"{st.session_state['stream_session_id']}:{row['id']}:{index}",
                        "id": str(row["id"]), "title": str(row["title"]),
                        "probability_fake": probability,
                        "decision": "review" if result.get("decision") == "review" else policy.decide(probability),
                        "threshold": float(policy.threshold), "reference_label": int(row["label"]),
                        "feedback_label": None, "latency_ms": round(latency_ms, 3),
                    }
                    streaming.append_event({"type": "prediction", **{key: value for key, value in event.items() if key != "reference_label"}})
                    events.append(event)
                    st.session_state["stream_cursor"] = index + 1
            st.rerun()
        except Exception as exc:
            st.error(f"Обработка остановлена: {exc}")
    if not events:
        return
    st.subheader("Лента решений")
    event_frame = pd.DataFrame(events)
    st.caption(f"Медианное время классификации в сессии: {event_frame['latency_ms'].median():.1f} мс на новость. Время интерфейса и записи журнала не включено.")
    shown = event_frame[["title", "probability_fake", "decision", "threshold", "feedback_label"]].copy()
    shown["decision"] = shown["decision"].map(DECISIONS)
    shown["feedback_label"] = shown["feedback_label"].map({0: "Истинная", 1: "Фейковая"}).fillna("Ожидает проверки")
    shown = shown.rename(columns={"title": "Заголовок", "probability_fake": "Оценка «фейк»", "decision": "Решение", "threshold": "Порог", "feedback_label": "Обратная связь"})
    st.dataframe(shown.iloc[::-1], hide_index=True, width="stretch", column_config={"Оценка «фейк»": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    st.subheader("Проверить решение")
    selected = st.selectbox(
        "Новость для обратной связи", list(range(len(events) - 1, -1, -1)),
        format_func=lambda index: events[index]["title"][:110], key="stream_selected_event",
    )
    event = events[selected]
    original = next((row for row in queue if str(row["id"]) == event["id"]), None)
    if original:
        with st.expander("Текст новости и исходная метка"):
            st.write(original["text"])
            st.caption(f"Метка в ISOT: {'фейковая' if event['reference_label'] else 'истинная'}. Она не поступает в адаптацию автоматически.")
    disabled = event.get("feedback_label") is not None
    a, b, c = st.columns(3)
    a.button("Моя оценка: истинная", on_click=stream_feedback, args=(selected, 0, "manual"), disabled=disabled, width="stretch")
    b.button("Моя оценка: фейковая", on_click=stream_feedback, args=(selected, 1, "manual"), disabled=disabled, width="stretch")
    c.button("Принять метку ISOT", on_click=stream_feedback, args=(selected, event["reference_label"], "dataset_confirmed"), disabled=disabled, width="stretch")
    st.caption("Только явно отправленные метки обновляют порог. Модель не переобучается; уже показанные решения сохраняются.")
    notice = st.session_state.get("stream_feedback_notice")
    if notice:
        feedback_messages = {
            "insufficient_feedback": "Метка принята. Для адаптации нужно не менее 12 подтверждённых примеров.",
            "duplicate_event": "Эта новость уже получила обратную связь.",
            "both_labels_required": "Для адаптации нужны подтверждённые примеры обоих классов.",
            "no_improvement": "Метка принята. Изменение порога не улучшает макро-F1 на окне обратной связи.",
            "confirmed_feedback_improvement": "Новый порог улучшает макро-F1 на окне подтверждённых меток.",
        }
        message = feedback_messages.get(notice.get("reason"), "Обратная связь принята.")
        if notice.get("changed"):
            st.success(f"Порог изменён: {notice.get('previous_threshold', 0):.2f} → {notice.get('threshold', 0):.2f}. {message}")
        else:
            st.info(message)
    if len(events) >= 20 and bundle.get("reference_scores") is not None:
        st.subheader("Изменение распределения оценок")
        try:
            drift = streaming.drift_summary(bundle["reference_scores"], event_frame["probability_fake"].to_numpy())
            a, b = st.columns([1, 2])
            a.metric("Статистика KS", f"{float(drift.get('ks_statistic', 0)):.3f}")
            b.caption(str(drift.get("interpretation", "")))
            st.line_chart(event_frame[["probability_fake", "threshold"]].rename(columns={"probability_fake": "Оценка «фейк»", "threshold": "Порог"}), color=["#cb8558", "#3e7860"])
            st.caption("Сравнение с validation. Сдвиг оценок сам по себе не доказывает ошибку и не запускает адаптацию.")
        except (ValueError, TypeError) as exc:
            st.caption(f"Для сравнения распределений пока недостаточно данных: {exc}")
    csv_download(event_frame.drop(columns=["reference_label"]), "stream_session.csv", "Скачать журнал сессии")


def methodological_audit(bundle):
    import plotly.graph_objects as go
    directory = Path(REPORTS_DIR) / "audit_v2"
    path = directory / "audit_report.json"
    st.title("Методологический аудит")
    if not path.exists():
        st.info("Новый аудит ещё рассчитывается. Результаты появятся после python scripts/audit_cycle.py.")
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    st.caption(f"Версия {report['version']} · protocol SHA-256 {report['protocol_sha256'][:16]}…")
    st.warning("Допуск: research_only. Статистические оценки и согласие моделей не заменяют независимую проверку фактов.")
    tabs = st.tabs(["Цикл и KPI", "До / после и raw data", "Порог и согласие", "ANOVA / SHAP", "Устойчивость", "Право и версия"])
    with tabs[0]:
        st.write(" → ".join(report["methodological_cycle"]))
        st.write("Цель: помощь ручной проверке англоязычных новостей. Главный контроль — независимый внешний корпус.")
        st.json(report["kpi"])
        st.dataframe(pd.DataFrame(report["partitions"]).T, width="stretch")
        st.caption(report["internal_test_status"])
        st.write(f"Дополнительных исключений между временными ролями: {report['contamination_exclusions']}.")
        st.caption("Признаки: только заголовок и текст. Метка, рубрика, дата, URL, tweet IDs и имя файла не подаются в модель.")
    with tabs[1]:
        rows = []
        for variant in ("before", "after"):
            for name, unit in (("internal_full_text", "ISOT / полный текст"), ("internal_headlines", "ISOT / заголовок"), ("external_raw_headlines", "FakeNewsNet / raw заголовок")):
                entry = report[variant][name]
                metric = entry["metrics"]
                rows.append({"Вариант": variant, "Проверка": unit, "N": metric["rows"],
                             "Macro-F1": metric["f1_macro"], "CI95 lower": metric["macro_f1_ci95"][0],
                             "CI95 upper": metric["macro_f1_ci95"][1], "Accuracy": metric["accuracy"],
                             "Brier": metric["brier"], "ECE": metric["ece_10_bins"],
                             "Auto coverage": entry["selective_policy"]["coverage"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Полные тексты и заголовки сравниваются отдельно. Внешние данные не использовались для настройки; их контрольные суммы и ревизия источника зафиксированы.")
        st.json(report["external_manifest"], expanded=False)
        for source, entry in report["external_by_source"].items():
            st.write(f"**{source}** · macro-F1 {entry['metrics']['f1_macro']:.3f} · N={entry['metrics']['rows']}")
        st.subheader("Матрица ошибок на внешних raw заголовках")
        matrix = report["after"]["external_raw_headlines"]["metrics"]["confusion_matrix"]
        st.dataframe(pd.DataFrame(matrix, index=["Метка real", "Метка fake"], columns=["Прогноз real", "Прогноз fake"]), width="stretch")
        for filename, label in (("after_internal_calibration.csv", "ISOT"), ("after_external_calibration.csv", "FakeNewsNet")):
            if (directory / filename).exists():
                curve = pd.read_csv(directory / filename)
                fig = go.Figure()
                fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="идеальная калибровка")
                fig.add_scatter(x=curve.mean_score, y=curve.fake_fraction, mode="lines+markers", name=label)
                fig.update_layout(title=f"Reliability diagram · {label}", xaxis_title="Оценка модели", yaxis_title="Доля fake по меткам корпуса", height=330)
                st.plotly_chart(fig, width="stretch")
    with tabs[2]:
        policy = report["after"]["policy"]
        st.json(policy)
        st.caption("Это зафиксированный порог решения, зона проверки и минимальное согласие. У каждого участника свой Platt-калибратор; их среднее откалибровано на другой выборке.")
        x = np.linspace(0, 1, 201)
        y = np.linspace(0, 1, 101)
        xx, yy = np.meshgrid(x, y)
        accepted = (np.abs(xx-policy["threshold"]) >= policy["margin"]) & (yy >= policy["consensus"]-1e-9) & policy["enabled"]
        z = np.where(accepted, np.where(xx >= policy["threshold"], 2, 0), 1)
        fig = go.Figure(go.Heatmap(x=x, y=y, z=z, zmin=0, zmax=2, colorscale=[[0, "#39735c"], [.49, "#e9e6d8"], [.51, "#e9e6d8"], [1, "#bf755b"]], colorbar={"tickvals": [0, 1, 2], "ticktext": ["real", "review", "fake"]}))
        fig.update_layout(height=370, xaxis_title="Откалиброванная оценка консенсуса", yaxis_title="Доля согласных участников")
        st.plotly_chart(fig, width="stretch")
        st.caption("Фактическая граница правила по оценке и согласию, не проекция пространства текстовых признаков. Пользовательские автоматические вердикты дополнительно закрыты release gate.")
        st.subheader("Согласие и согласованные ошибки на внешних данных")
        agreement = report["after"]["external_raw_headlines"]["agreement"]
        st.dataframe(pd.DataFrame(agreement["pairwise"]), hide_index=True, width="stretch")
        st.write(f"Ошибок среди единогласных решений: {percent(agreement['unanimous_wrong_fraction'])}.")
        st.json(report["boundary_stability"])
    with tabs[3]:
        st.write("ANOVA и SHAP анализируют ту же корпусную метку и ту же модель. Их совпадение не является независимой проверкой правильности.")
        circularity = report["circularity"]
        st.write(f"Пересечение top-100 ANOVA для метки и Reuters-proxy: {circularity['top100_ANOVA_label_source_overlap']}.")
        st.json(circularity["component_shap"])
        st.dataframe(pd.DataFrame(circularity["negative_controls"]), hide_index=True, width="stretch")
        st.caption(f"Перестановочный p-value: {circularity['permutation_p_value']:.3f}; разрешение {circularity['permutation_resolution']:.3f}. Отбор ANOVA повторяется только на train при каждой перестановке.")
        if (directory / "global_xai_anova.csv").exists():
            global_xai = pd.read_csv(directory / "global_xai_anova.csv")
            st.bar_chart(global_xai.head(20).set_index("feature")[["global_mean_abs_component_SHAP"]], horizontal=True)
            st.caption("SHAP: фиксированный train-background, calibrated primary logit; корреляции биграмм/униграмм не делают атрибуции причинными.")
        for filename, label in (("local_component_shap.csv", "Локальный SHAP участника"), ("local_consensus_deletions.csv", "Локальные удаления слов и оценка всего консенсуса")):
            if (directory / filename).exists():
                with st.expander(label):
                    st.dataframe(pd.read_csv(directory / filename), hide_index=True, width="stretch")
    with tabs[4]:
        st.dataframe(pd.DataFrame(report["robustness"]["results"]), hide_index=True, width="stretch")
        st.caption(report["robustness"]["limitation"])
        st.json(report["boundary_stability"])
        st.write(f"Признаков из заранее объявленного списка удалено: {report['removed_feature_count']}. Список не подбирался по внешнему test.")
        if (directory / "removed_bad_features.csv").exists():
            st.dataframe(pd.read_csv(directory / "removed_bad_features.csv").head(100), hide_index=True, width="stretch")
    with tabs[5]:
        st.json(report["legal"], expanded=False)
        st.json(report["release"])
        st.write("Контрольная сумма параметров до/после внешней проверки совпадает:", report["freeze_verified"])
        st.code("python scripts/verify_version.py", language="powershell")
        st.caption("Версия модели, снимок исходников, протокол, внешние CSV и отчёт имеют отдельные SHA-256. Повторное использование внешнего набора отмечается; оно не становится новой независимой проверкой.")
    st.download_button("Скачать аудит JSON", json.dumps(report, ensure_ascii=False, indent=2), file_name="audit_v2.json")


articles, bundle, summary, resource_errors = resources()
with st.sidebar:
    st.markdown('<div class="eyebrow">NEWS / 15</div>', unsafe_allow_html=True)
    st.markdown("### Лаборатория новостей")
    st.caption("От данных к объяснимому решению")
    section = st.radio("Раздел", SECTIONS, label_visibility="collapsed")
    st.divider()
    st.caption("Модель готова" if bundle else "Модель ещё не обучена")
    if not articles.empty:
        st.caption(f"ISOT · {integer(len(articles))} новостей")
    st.caption("English model · русский интерфейс")

if section == "Обзор":
    overview(articles, bundle, summary)
elif section == "Проверка новости":
    check_article(articles, bundle)
elif section == "Модели и данные":
    models_and_data(articles, bundle, summary, resource_errors)
elif section == "Эксперименты":
    experiment_page(articles, bundle)
elif section == "Поток и обратная связь":
    stream_page(articles, bundle)
else:
    methodological_audit(bundle)
