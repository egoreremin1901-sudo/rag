from statistics import mean

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    BASELINE_MAX_INPUT_TOKENS,
    BASELINE_MAX_NEW_TOKENS,
    BASELINE_MODEL_NAME,
    GROUND_TRUTH_PATH,
    METRICS_DIR,
    QUESTIONS_PATH,
)
from utils import calculate_metrics, judge_with_groq, read_json, save_json


def load_questions() -> list[dict]:
    """Загружает вопросы из questions.json."""
    return read_json(QUESTIONS_PATH)


def load_ground_truth() -> dict[str, str]:
    """Загружает правильные ответы и делает словарь: id вопроса -> правильный ответ."""
    rows = read_json(GROUND_TRUTH_PATH)
    return {row["id"]: row["answer"] for row in rows}


def load_local_model() -> tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    """Загружает маленькую локальную модель Hugging Face."""
    if torch.cuda.is_available():
        device = "cuda"
        model_dtype = torch.float16
    else:
        device = "cpu"
        model_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        BASELINE_MODEL_NAME,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    return tokenizer, model, device


def make_prompt_without_rag(question: str, tokenizer: AutoTokenizer) -> str:
    """Создает prompt без контекста. Это baseline: модель отвечает только из своих знаний."""
    messages = [
        {
            "role": "system",
            "content": "Ты помощник, который кратко отвечает на русском языке.",
        },
        {
            "role": "user",
            "content": f"Ответь на вопрос кратко.\n\nВопрос: {question}",
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_answer(
    question: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
) -> str:
    """Передает вопрос в модель и возвращает текст ответа."""
    prompt = make_prompt_without_rag(question, tokenizer)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=BASELINE_MAX_INPUT_TOKENS,
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=BASELINE_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_answers(predictions: list[dict], ground_truth: dict[str, str]) -> dict:
    """Сравнивает все ответы модели с ground_truth и считает средние метрики."""
    rows = []

    for item in tqdm(predictions, desc="Evaluate baseline"):
        target = ground_truth[item["id"]]
        metrics = calculate_metrics(item["prediction"], target)
        judge = judge_with_groq(item["question"], item["prediction"], target)

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "target": target,
                "prediction": item["prediction"],
                "metrics": metrics,
                "judge": judge,
            }
        )

    judge_scores = [row["judge"]["score"] for row in rows if row["judge"] is not None]
    summary = {
        "exact_match": mean(row["metrics"]["exact_match"] for row in rows),
        "bleu": mean(row["metrics"]["bleu"] for row in rows),
        "rouge_l": mean(row["metrics"]["rouge_l"] for row in rows),
        "llm_as_judge": mean(judge_scores) if judge_scores else None,
    }

    return {"run_name": "llm_without_rag", "summary": summary, "rows": rows}


def main() -> None:
    """Главный порядок запуска baseline без RAG."""
    questions = load_questions()
    ground_truth = load_ground_truth()
    tokenizer, model, device = load_local_model()

    predictions = []
    for item in tqdm(questions, desc="Generate baseline answers"):
        answer = generate_answer(item["question"], tokenizer, model, device)
        predictions.append(
            {
                "id": item["id"],
                "question": item["question"],
                "prediction": answer,
            }
        )

    metrics = evaluate_answers(predictions, ground_truth)

    save_json(predictions, METRICS_DIR / "01_llm_without_rag_predictions.json")
    save_json(metrics, METRICS_DIR / "01_llm_without_rag_metrics.json")

    print("Baseline без RAG готов.")
    print("Ответы: metrics/01_llm_without_rag_predictions.json")
    print("Метрики: metrics/01_llm_without_rag_metrics.json")


if __name__ == "__main__":
    main()
