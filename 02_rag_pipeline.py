import shutil
from statistics import mean

import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    ARTICLES_PATH,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    GROUND_TRUTH_PATH,
    METRICS_DIR,
    QUESTIONS_PATH,
    RAG_MAX_INPUT_TOKENS,
    RAG_MAX_NEW_TOKENS,
    RAG_MODEL_NAME,
    RETRIEVER_K,
)
from utils import calculate_metrics, judge_with_groq, read_json, save_json, save_metrics_table


def load_articles() -> list[dict]:
    """Загружает статьи, из которых будем делать базу знаний."""
    return read_json(ARTICLES_PATH)


def load_questions() -> list[dict]:
    """Загружает вопросы для проверки RAG."""
    return read_json(QUESTIONS_PATH)


def load_ground_truth() -> dict[str, str]:
    """Загружает правильные ответы и делает словарь: id вопроса -> правильный ответ."""
    rows = read_json(GROUND_TRUTH_PATH)
    return {row["id"]: row["answer"] for row in rows}


def make_langchain_documents(articles: list[dict]) -> list[Document]:
    """Превращает наши статьи в документы LangChain."""
    documents = []

    for article in articles:
        text = f"{article['title']}\n\n{article['text']}"
        document = Document(
            page_content=text,
            metadata={"article_id": article["id"], "title": article["title"]},
        )
        documents.append(document)

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Режет большие документы на чанки, чтобы retriever искал не по всей статье сразу."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_chroma_index(chunks: list[Document]) -> Chroma:
    """Создает Chroma-индекс: текстовые чанки превращаются в embeddings и сохраняются."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def find_context(question: str, vector_store: Chroma) -> tuple[str, list[dict]]:
    """Ищет в Chroma несколько чанков, которые похожи на вопрос."""
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
    found_docs = retriever.invoke(question)

    context_parts = []
    sources = []

    for doc in found_docs:
        article_id = doc.metadata["article_id"]
        title = doc.metadata["title"]
        context_parts.append(f"Статья {article_id}: {doc.page_content}")
        sources.append({"article_id": article_id, "title": title})

    return "\n\n".join(context_parts), sources


def load_local_model() -> tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    """Загружает ту же локальную модель, что и в baseline."""
    if torch.cuda.is_available():
        device = "cuda"
        model_dtype = torch.float16
    else:
        device = "cpu"
        model_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(RAG_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        RAG_MODEL_NAME,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    return tokenizer, model, device


def make_prompt_with_rag(question: str, context: str, tokenizer: AutoTokenizer) -> str:
    """Создает prompt для RAG: теперь модель видит вопрос и найденный контекст."""
    messages = [
        {
            "role": "system",
            "content": "Ты помощник, который отвечает только по переданному контексту.",
        },
        {
            "role": "user",
            "content": (
                "Ответь кратко и по-русски. Используй только контекст.\n\n"
                f"Контекст:\n{context}\n\n"
                f"Вопрос: {question}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_answer_with_rag(
    question: str,
    context: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
) -> str:
    """Генерирует ответ локальной моделью, но уже с найденным RAG-контекстом."""
    prompt = make_prompt_with_rag(question, context, tokenizer)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=RAG_MAX_INPUT_TOKENS,
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=RAG_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_answers(predictions: list[dict], ground_truth: dict[str, str]) -> dict:
    """Сравнивает RAG-ответы с правильными ответами."""
    rows = []

    for item in tqdm(predictions, desc="Evaluate RAG"):
        target = ground_truth[item["id"]]
        metrics = calculate_metrics(item["prediction"], target)
        judge = judge_with_groq(item["question"], item["prediction"], target)

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "target": target,
                "prediction": item["prediction"],
                "sources": item["sources"],
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

    return {"run_name": "rag_chroma", "summary": summary, "rows": rows}


def main() -> None:
    """Главный порядок запуска RAG-пайплайна."""
    articles = load_articles()
    questions = load_questions()
    ground_truth = load_ground_truth()

    documents = make_langchain_documents(articles)
    chunks = split_documents(documents)
    vector_store = build_chroma_index(chunks)

    tokenizer, model, device = load_local_model()

    predictions = []
    for item in tqdm(questions, desc="Generate RAG answers"):
        context, sources = find_context(item["question"], vector_store)
        answer = generate_answer_with_rag(item["question"], context, tokenizer, model, device)
        predictions.append(
            {
                "id": item["id"],
                "question": item["question"],
                "prediction": answer,
                "sources": sources,
            }
        )

    metrics = evaluate_answers(predictions, ground_truth)

    save_json(predictions, METRICS_DIR / "02_rag_predictions.json")
    save_json(metrics, METRICS_DIR / "02_rag_metrics.json")
    save_metrics_table(METRICS_DIR)

    print("RAG готов.")
    print("Ответы: metrics/02_rag_predictions.json")
    print("Метрики: metrics/02_rag_metrics.json")
    print("Общая таблица: metrics/summary.md")


if __name__ == "__main__":
    main()
