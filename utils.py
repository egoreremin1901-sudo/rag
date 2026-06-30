import json
import os
import re
from pathlib import Path

from groq import DefaultHttpxClient, Groq
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

from config import GROQ_JUDGE_MODEL

_groq_error_was_shown = False


def read_json(path: Path) -> list[dict]:
    """Читает json-файл и возвращает список словарей."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict | list[dict], path: Path) -> None:
    """Сохраняет результат в json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def format_score(value: float | None) -> str:
    """Форматирует метрику для markdown-таблицы."""
    if value is None:
        return "skipped"
    return f"{value:.4f}"


def save_metrics_table(metrics_dir: Path) -> None:
    """Создает простую таблицу с общими метриками по всем запускам."""
    metric_files = [
        metrics_dir / "01_llm_without_rag_metrics.json",
        metrics_dir / "02_rag_metrics.json",
    ]

    rows = []
    for path in metric_files:
        if path.exists():
            rows.append(read_json(path))

    if not rows:
        return

    lines = [
        "# Общие метрики",
        "",
        "| Запуск | Exact match | BLEU | ROUGE-L | LLM as judge |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        summary = row["summary"]
        lines.append(
            "| {run} | {exact} | {bleu} | {rouge} | {judge} |".format(
                run=row["run_name"],
                exact=format_score(summary["exact_match"]),
                bleu=format_score(summary["bleu"]),
                rouge=format_score(summary["rouge_l"]),
                judge=format_score(summary["llm_as_judge"]),
            )
        )

    (metrics_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_text(text: str) -> str:
    """Упрощает текст перед сравнением."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9%.,:-]+", " ", text)
    return " ".join(text.split())


def calculate_metrics(prediction: str, target: str) -> dict:
    """Считает BLEU и ROUGE-L между ответом модели и правильным ответом."""
    bleu = BLEU(effective_order=True)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    normalized_prediction = normalize_text(prediction)
    normalized_target = normalize_text(target)

    return {
        "exact_match": float(normalized_prediction == normalized_target),
        "bleu": bleu.sentence_score(prediction, [target]).score / 100,
        "rouge_l": rouge.score(target, prediction)["rougeL"].fmeasure,
    }


def judge_with_groq(question: str, prediction: str, target: str) -> dict | None:
    """Просит Groq-модель оценить ответ. Это llm_as_judge."""
    global _groq_error_was_shown

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        if not _groq_error_was_shown:
            print("Groq judge skipped: GROQ_API_KEY не найден в .env")
            _groq_error_was_shown = True
        return None

    client = Groq(
        api_key=api_key,
        http_client=DefaultHttpxClient(trust_env=False),
    )
    prompt = f"""
Оцени ответ модели по эталону.

Верни только JSON:
{{"score": 0.0, "reason": "короткая причина"}}

score:
1.0 - ответ верный
0.5 - ответ частично верный
0.0 - ответ неверный

Вопрос: {question}
Эталон: {target}
Ответ модели: {prediction}
""".strip()

    try:
        response = client.chat.completions.create(
            model=GROQ_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=160,
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
    except Exception as error:
        if not _groq_error_was_shown:
            print(f"Groq judge skipped: {error}")
            _groq_error_was_shown = True
        return None

    score = float(data.get("score", 0.0))
    return {"score": max(0.0, min(1.0, score)), "reason": data.get("reason", "")}
