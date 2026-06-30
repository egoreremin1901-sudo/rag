 RAG pipeline

Это учебный проект.

Главная идея проекта:

```text
Вопрос -> LLM без базы знаний -> ответ -> метрики

Вопрос -> поиск по базе знаний -> найденный контекст -> LLM -> ответ -> метрики
```

После этого сравниваем, стало ли лучше с RAG.

## Установка и запуск

Создать виртуальное окружение:

```bash
python -m venv .venv
```

Активировать на Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Сначала лучше сбросить прокси-переменные для `pip`, если они мешают установке:

```powershell
$env:ALL_PROXY=""
$env:all_proxy=""
$env:HTTP_PROXY=""
$env:http_proxy=""
$env:HTTPS_PROXY=""
$env:https_proxy=""
```

Обновить `pip`:

```powershell
python -m pip install --upgrade pip
```

Поставить PyTorch отдельно с официального CUDA-индекса:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Поставить остальные зависимости:

```powershell
pip install -r requirements.txt
```

В `requirements.txt` лежат остальные библиотеки проекта. PyTorch ставится
отдельно, потому что у него свой индекс для CUDA-версий.

Если установка через CUDA-индекс не работает, можно временно поставить CPU-версию:

```powershell
pip install torch
pip install -r requirements.txt
```

Проект запустится и на CPU, просто генерация будет намного медленнее.
Для `llm_as_judge` нужен Groq API key. В проекте уже есть файл:

```text
.env
```

Открой `.env` и вставь ключ:

```text
GROQ_API_KEY=твой_ключ
```

Если ключа нет, проект все равно запустится. Просто метрика `llm_as_judge`
будет пропущена.

В `.env` также лежат имена моделей:

```text
BASELINE_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
RAG_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
GROQ_JUDGE_MODEL=llama-3.3-70b-versatile
```

Запускать проект нужно по порядку.

Сначала baseline без RAG:

```bash
python 01_llm_without_rag.py
```

Потом RAG:

```bash
python 02_rag_pipeline.py
```

После запуска результаты будут здесь:

```text
metrics/01_llm_without_rag_predictions.json
metrics/01_llm_without_rag_metrics.json
metrics/02_rag_predictions.json
metrics/02_rag_metrics.json
metrics/comparison.md
```

Главный файл для сравнения:

```text
metrics/comparison.md
```

Папка `metrics/` не добавлена в `.gitignore`, поэтому после прогона ее можно
закоммитить и отправить на GitHub вместе с результатами.

## Главные файлы проекта

1. `01_llm_without_rag.py` - сначала проверяем обычную LLM без RAG.
2. `02_rag_pipeline.py` - потом строим RAG: индексируем статьи, ищем контекст и отвечаем с ним.
3. `config.py` - все настройки проекта: пути, модели, размеры чанков, лимиты токенов.
4. `utils.py` - общие функции, которые нужны и baseline, и RAG.



функции вроде `read_json()`, `save_json()`, `calculate_metrics()` и
`judge_with_groq()` нужны сразу в двух местах:

```text
01_llm_without_rag.py
02_rag_pipeline.py
```

В этом проекте общий файл называется:

```text
utils.py
```

А в других файлах он подключается так:

```python
from utils import calculate_metrics, judge_with_groq, read_json, save_json config.py
```

 настройки вынесены в:

```text
config.py
```

В нем есть отдельные блоки.

Общие пути:

```python
QUESTIONS_PATH
GROUND_TRUTH_PATH
METRICS_DIR
```

Настройки для `01_llm_without_rag.py`:

```python
BASELINE_MODEL_NAME
BASELINE_MAX_INPUT_TOKENS
BASELINE_MAX_NEW_TOKENS
```

Настройки для `02_rag_pipeline.py`:

```python
ARTICLES_PATH
CHROMA_DIR
RAG_MODEL_NAME
EMBEDDING_MODEL_NAME
CHUNK_SIZE
CHUNK_OVERLAP
RETRIEVER_K
RAG_MAX_INPUT_TOKENS
RAG_MAX_NEW_TOKENS
```

Настройки для Groq judge:

```python
GROQ_JUDGE_MODEL
```

В рабочих файлах это подключается через импорт:

```python
from config import QUESTIONS_PATH, METRICS_DIR
```



## Какие данные есть в проекте

В проекте лежат 3 файла с данными:

```text
articles.json          - 10 статей корпоративной базы знаний
questions.json         - 50 вопросов по этим статьям
ground_truth.json      - правильные ответы на вопросы
```

`articles.json` нужен для RAG. Из него мы строим базу знаний.

`questions.json` нужен для тестов. По нему мы прогоняем модель.

`ground_truth.json` нужен для оценки. С ним сравниваем ответы модели.

## Почему выбрана такая модель

По умолчанию используется:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Почему именно она:

- она небольшая, примерно 1.5B параметров;
- она instruction-модель, то есть умеет отвечать на вопросы;
- она нормально работает с русским языком;

Для поиска по тексту используется embedding-модель:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Она нужна не для генерации ответов, а для поиска похожих кусков текста.  
Она тоже небольшая и поддерживает русский язык.

## Файл 1: LLM без RAG

Файл:

```text
01_llm_without_rag.py
```

Запуск:

```bash
python 01_llm_without_rag.py
```

Это baseline. Baseline нужен, чтобы понять, как модель отвечает без базы знаний.
То есть мы не даем ей статьи из `articles.json`.

### Порядок работы baseline

Когда запускается файл, Python доходит до конца и видит:

```python
if __name__ == "__main__":
    main()
```

Значит первой реально запускается функция:

```python
main()
```

Внутри `main()` порядок такой:

```text
1. load_questions()
2. load_ground_truth()
3. load_local_model()
4. generate_answer() для каждого вопроса
5. evaluate_answers()
6. save_json()
```



### Что делает каждая функция в baseline-файле

`load_questions()`

Загружает список вопросов из `questions.json`.

Внутри использует `read_json()` из `utils.py`.

`load_ground_truth()`

Загружает правильные ответы из `ground_truth.json`.
На выходе получается словарь:

```python
{
    "q1": "правильный ответ",
    "q2": "правильный ответ"
}
```

Так удобнее быстро доставать правильный ответ по id вопроса.

Внутри тоже использует `read_json()` из `utils.py`.

### Что приходит из [utils.py](http://utils.py)

В baseline-файле есть импорт:

```python
from utils import calculate_metrics, judge_with_groq, read_json, save_json
```

Это значит:

- `read_json()` читает JSON-файлы;
- `save_json()` сохраняет результаты;
- `calculate_metrics()` считает `exact_match`, `BLEU`, `ROUGE-L`;
- `judge_with_groq()` делает `llm_as_judge`.

Эти функции вынесены отдельно, потому что они нужны и в baseline, и в RAG.

`load_local_model()`

Загружает tokenizer и модель из Hugging Face.

Tokenizer превращает текст в токены.  
Модель получает токены и генерирует новые токены.  
Потом tokenizer превращает токены обратно в текст.

`make_prompt_without_rag(question, tokenizer)`

Создает prompt для модели без RAG.

Тут нет контекста из статей. Есть только вопрос.
Поэтому модель отвечает из своих внутренних знаний.

`generate_answer(question, tokenizer, model, device)`

Это функция инференса.

Она:

```text
1. Делает prompt.
2. Токенизирует prompt.
3. Передает токены в model.generate().
4. Декодирует новые токены обратно в текст.
```

`calculate_metrics(prediction, target)` из `utils.py`

Считает метрики между ответом модели и правильным ответом:

- `exact_match` - полное совпадение после нормализации;
- `bleu` - насколько похожа формулировка ответа;
- `rouge_l` - насколько хорошо ответ покрывает правильный ответ.

`judge_with_groq(question, prediction, target)` из `utils.py`

Это `llm_as_judge`.

Мы отправляем в Groq:

- вопрос;
- правильный ответ;
- ответ нашей модели.

Groq возвращает оценку:

```text
1.0 - верно
0.5 - частично верно
0.0 - неверно
```

`evaluate_answers(predictions, ground_truth)`

Проходит по всем ответам модели и считает метрики.
В конце считает среднее значение по всем вопросам.

`main()`

Главная функция baseline. Она связывает все шаги вместе.

### Что сохраняется после baseline

После запуска появятся файлы:

```text
metrics/01_llm_without_rag_predictions.json
metrics/01_llm_without_rag_metrics.json
```

Первый файл - все ответы модели.

Второй файл - ответы, правильные ответы, метрики и judge-оценки.

## Файл 2: RAG pipeline

Файл:

```text
02_rag_pipeline.py
```

Запуск:

```bash
python 02_rag_pipeline.py
```

RAG означает Retrieval-Augmented Generation.

По-простому:

```text
сначала ищем нужный текст в базе знаний,
потом передаем найденный текст в LLM,
потом LLM отвечает уже не из головы, а по контексту.
```



### Порядок работы RAG

Как и в первом файле, первым запускается:

```python
main()
```

Внутри `main()` порядок такой:

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
12. save_comparison()
```



### Что делает каждая функция RAG

`load_articles()`

Загружает статьи из `articles.json`.
Это наша маленькая база знаний.

`make_langchain_documents(articles)`

LangChain работает с объектами `Document`.

Каждая статья превращается в такой объект:

```python
Document(
    page_content="заголовок + текст статьи",
    metadata={"article_id": "...", "title": "..."}
)
```

`page_content` - текст, по которому будем искать.

`metadata` - служебная информация. Она нужна, чтобы потом понять,
из какой статьи был найден контекст.

`split_documents(documents)`

Режет статьи на чанки.

Почему нельзя всегда искать по всей статье целиком:

- большие тексты хуже искать;
- в prompt не хочется передавать лишний текст;
- RAG обычно работает именно с кусками документов.

В проекте стоит:

```python
chunk_size=700
chunk_overlap=120
```

`chunk_size` - примерный размер одного чанка.

`chunk_overlap` - небольшой повтор между соседними чанками.
Он нужен, чтобы важная мысль не потерялась на границе двух чанков.

`build_chroma_index(chunks)`

Это этап индексации.

Он делает главное:

```text
текстовые чанки -> embeddings -> Chroma database
```

Embedding - это числовой вектор текста.
Похожие по смыслу тексты должны иметь похожие векторы.

Chroma хранит эти векторы и умеет быстро искать похожие.

`find_context(question, vector_store)`

Это этап retrieval.

Функция берет вопрос и ищет в Chroma 3 самых похожих чанка:

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```

`k=3` означает: вернуть 3 найденных куска текста.

На выходе функция возвращает:

```text
context - найденный текст для prompt
sources - список статей, откуда взяли контекст
```

`make_prompt_with_rag(question, context, tokenizer)`

Создает prompt уже с контекстом.

То есть модель получает:

```text
Контекст:
...

Вопрос:
...
```

И мы просим отвечать только по контексту.

`generate_answer_with_rag(question, context, tokenizer, model, device)`

Это инференс RAG.

Отличие от baseline только одно:

```text
baseline: вопрос -> модель
RAG: вопрос + найденный контекст -> модель
```

`calculate_metrics(...)`, `judge_with_groq(...)`, `evaluate_answers(...)`

Работают почти так же, как в первом файле.
Они снова сравнивают ответы модели с `ground_truth.json`.

`save_comparison(rag_metrics)`

Создает удобный файл:

```text
metrics/comparison.md
```

Там будет таблица сравнения:

```text
llm_without_rag
rag_chroma
```



## Как запускать весь проект по порядку

Сначала baseline:

```bash
python 01_llm_without_rag.py
```

Потом RAG:

```bash
python 02_rag_pipeline.py
```

Потом открыть:

```text
metrics/comparison.md
```



## Какие результаты будут созданы

После первого файла:

```text
metrics/01_llm_without_rag_predictions.json
metrics/01_llm_without_rag_metrics.json
```

После второго файла:

```text
metrics/02_rag_predictions.json
metrics/02_rag_metrics.json
metrics/comparison.md
```

Также появится папка:

```text
chroma_db/
```

Это локальная база Chroma с индексом статей.

Baseline pipeline:

```text
questions -> prompt -> local LLM -> answer -> metrics files
```

RAG pipeline:

```text
articles -> chunks -> embeddings -> Chroma
question -> Chroma search -> context -> prompt -> local LLM -> answer -> metrics files
```



## Проверка кода

Форматирование:

```bash
ruff format .
```

Линтер:

```bash
ruff check .
```

Проверка синтаксиса:

```bash
python -m compileall 01_llm_without_rag.py 02_rag_pipeline.py config.py utils.py
```



## Что можно улучшить потом

Это специально простой baseline RAG. Потом можно улучшать:

- менять `chunk_size`;
- менять `chunk_overlap`;
- менять `k` в retriever;
- добавить reranker;
- добавить query rewriting;
- попробовать другую локальную модель;
- сделать отдельные красивые отчеты.

Но первый учебный вариант лучше держать простым:

```text
load data -> index data -> retrieve context -> generate answer -> evaluate
```

