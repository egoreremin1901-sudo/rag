 RAG pipeline

Это учебный проект: сначала проверяем обычную LLM без RAG, потом строим простой RAG и сравниваем метрики.

Главная идея проекта:

```text
Вопрос -> LLM без базы знаний -> ответ -> метрики

Вопрос -> поиск по базе знаний -> найденный контекст -> LLM -> ответ -> метрики
```

После этого смотрим, стало ли лучше с RAG.

## Установка и запуск

Создать виртуальное окружение:

```powershell
python -m venv .venv
```

Активировать окружение на Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Обновить `pip`:

```powershell
python -m pip install --upgrade pip
```

Поставить PyTorch. Если есть NVIDIA GPU и нужна CUDA-версия:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Если CUDA-версия не ставится, можно поставить обычную CPU-версию:

```powershell
pip install torch
```

Поставить остальные зависимости:

```powershell
pip install -r requirements.txt
```

Открыть файл `.env` и вставить Groq API key:

```text
GROQ_API_KEY=твой_ключ
```

В `.env` также лежат имена моделей:

```text
BASELINE_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
RAG_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
GROQ_JUDGE_MODEL=llama-3.3-70b-versatile
```

Если `GROQ_API_KEY` не заполнен, проект все равно запустится, но метрика `llm_as_judge` будет пропущена.

Запустить baseline без RAG:

```powershell
python 01_llm_without_rag.py
```

Запустить RAG:

```powershell
python 02_rag_pipeline.py
```

Главный файл с общим сравнением:

```text
metrics/summary.md
```

После запуска также появятся подробные JSON-файлы:

```text
metrics/01_llm_without_rag_predictions.json
metrics/01_llm_without_rag_metrics.json
metrics/02_rag_predictions.json
metrics/02_rag_metrics.json
```

Папка `metrics/` не добавлена в `.gitignore`, поэтому ее можно закоммитить и отправить на GitHub вместе с результатами.

## Главные файлы проекта

`01_llm_without_rag.py` - baseline. Модель отвечает на вопросы без статей и без поиска по базе знаний.

`02_rag_pipeline.py` - RAG pipeline. Скрипт индексирует статьи, ищет подходящий контекст и передает его модели.

`config.py` - настройки проекта: пути к файлам, имена моделей, размеры чанков, лимиты токенов.

`utils.py` - общие функции: чтение JSON, сохранение JSON, расчет метрик, Groq judge, сохранение таблицы метрик.

`articles.json` - 10 статей корпоративной базы знаний.

`questions.json` - 50 вопросов по статьям.

`ground_truth.json` - правильные ответы на вопросы.

## Почему выбрана такая модель

По умолчанию используется:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Причины:

- модель небольшая, около 1.5B параметров;
- ее реально запустить локально;
- она instruction-модель, то есть умеет отвечать на вопросы;
- она нормально работает с русским языком;
- для первого RAG важнее понять pipeline, а не брать самую большую модель.

Для поиска по тексту используется embedding-модель:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Она превращает текст в векторы, чтобы Chroma могла искать похожие по смыслу куски статей. Эта модель небольшая и поддерживает русский язык.

## Baseline Без RAG

Файл:

```text
01_llm_without_rag.py
```

Первая запускаемая функция:

```python
main()
```

Порядок внутри `main()`:

```text
1. load_questions()
2. load_ground_truth()
3. load_local_model()
4. generate_answer() для каждого вопроса
5. evaluate_answers()
6. save_json()
7. save_metrics_table()
```

Что происходит:

`load_questions()` читает вопросы из `questions.json`.

`load_ground_truth()` читает правильные ответы из `ground_truth.json` и делает словарь вида `id вопроса -> правильный ответ`.

`load_local_model()` загружает tokenizer и модель из Hugging Face. Если доступна CUDA, модель загружается на GPU в `torch.float16`, иначе на CPU в `torch.float32`.

`make_prompt_without_rag()` делает prompt только с вопросом. Здесь нет контекста из статей.

`generate_answer()` передает prompt в модель и получает ответ.

`evaluate_answers()` сравнивает ответы модели с `ground_truth.json`.

Результаты baseline:

```text
metrics/01_llm_without_rag_predictions.json
metrics/01_llm_without_rag_metrics.json
```

`predictions.json` нужен, чтобы посмотреть сырые ответы модели.

`metrics.json` нужен, чтобы посмотреть подробные метрики по каждому вопросу.

## RAG Pipeline

Файл:

```text
02_rag_pipeline.py
```

Первая запускаемая функция:

```python
main()
```

Порядок внутри `main()`:

```text
1. load_articles()
2. load_questions()
3. load_ground_truth()
4. make_langchain_documents()
5. split_documents()
6. build_chroma_index()
7. load_local_model()
8. find_context() для каждого вопроса
9. generate_answer_with_rag() для каждого вопроса
10. evaluate_answers()
11. save_json()
12. save_metrics_table()
```

Что происходит:

`load_articles()` читает статьи из `articles.json`.

`make_langchain_documents()` превращает статьи в объекты `Document`. В `page_content` лежит текст статьи, а в `metadata` лежат `article_id` и `title`.

`split_documents()` режет статьи на чанки. Сейчас используются `CHUNK_SIZE=700` и `CHUNK_OVERLAP=120`.

`build_chroma_index()` создает Chroma-индекс: чанки превращаются в embeddings и сохраняются в локальную базу `chroma_db/`.

`find_context()` ищет в Chroma несколько чанков, похожих на вопрос. Сейчас `RETRIEVER_K=3`, то есть возвращаются 3 найденных куска.

`make_prompt_with_rag()` делает prompt из найденного контекста и вопроса.

`generate_answer_with_rag()` передает prompt в модель и получает ответ уже с учетом найденного контекста.

`evaluate_answers()` считает метрики так же, как в baseline.

Результаты RAG:

```text
metrics/02_rag_predictions.json
metrics/02_rag_metrics.json
metrics/summary.md
```

В `02_rag_predictions.json` есть поле `sources`. Это список статей, из которых retriever взял контекст. По нему удобно проверять, нашел ли RAG правильную статью.

## Проверка кода

Форматирование:

```powershell
ruff format .
```

Линтер:

```powershell
ruff check .
```

Проверка синтаксиса:

```powershell
python -m compileall 01_llm_without_rag.py 02_rag_pipeline.py config.py utils.py
```

## Что Значат Метрики

`Exact match` - доля ответов, которые полностью совпали с эталоном после простой нормализации текста. Для генеративных моделей эта метрика обычно низкая, потому что модель может ответить правильно, но другими словами.

`BLEU` - метрика похожести формулировки ответа на эталон. Чем больше пересекающихся слов и фраз, тем выше BLEU. Для коротких QA-ответов BLEU полезен как грубый ориентир, но не всегда отражает смысл.

`ROUGE-L` - метрика, которая смотрит на самую длинную общую подпоследовательность между ответом модели и эталоном. Она показывает, насколько ответ покрывает правильный ответ похожими словами.

`LLM as judge` - оценка ответа другой LLM через Groq. В этом проекте judge получает вопрос, эталон и ответ модели, а потом ставит `1.0`, `0.5` или `0.0`. Это самая смысловая метрика из текущих, потому что она может засчитать ответ, написанный другими словами.

`skipped` в `LLM as judge` означает, что judge-оценка не была посчитана. Обычно причина в пустом или недоступном `GROQ_API_KEY`.

## Что Можно Улучшить Потом

Это специально простой baseline RAG. Потом можно пробовать:

- менять `CHUNK_SIZE`;
- менять `CHUNK_OVERLAP`;
- менять `RETRIEVER_K`;
- добавить reranker;
- добавить query rewriting;
- попробовать другую локальную модель;
- сделать более красивый отчет.

Первый учебный вариант лучше держать простым:

```text
load data -> index data -> retrieve context -> generate answer -> evaluate
```
