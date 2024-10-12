import logging
import re
import os
import json
import asyncio
import openai
import pandas as pd
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import io
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    PreCheckoutQueryHandler,
)

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация Firebase
firebase_credentials_str = os.environ.get('FIREBASE_CREDENTIALS', '{}')
try:
    firebase_credentials = json.loads(firebase_credentials_str)
    if not firebase_credentials:
        logger.error("Переменная окружения FIREBASE_CREDENTIALS пуста или содержит неверные данные.")
        raise ValueError("Неверные учетные данные Firebase.")
    
    firebase_admin.initialize_app(credentials.Certificate(firebase_credentials), {
        'databaseURL': os.environ.get('FIREBASE_DATABASE_URL')
    })
    logger.info("Firebase Admin SDK успешно инициализирован.")
except Exception as e:
    logger.exception("Ошибка при инициализации Firebase Admin SDK.")
    # В зависимости от вашей логики, вы можете завершить работу бота или предпринять другие действия
    # Например:
    # exit(1)

# Получение ссылки на корень базы данных
ref = db.reference('/')

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TOKEN")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment variables")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not set in environment variables")

openai.api_key = OPENAI_API_KEY

CATEGORIES_MODEL_NAME = "gpt-4o"
ANALYSIS_MODEL_NAME = "gpt-4o-mini"

# Определяем состояния разговора
(
    UPLOAD,
    QUESTION,
    COLUMN,
    CATEGORIES,
    PROCESSING,
    CHOOSE_CATEGORIES,
    EDIT_CATEGORIES,
    RARE_CATEGORIES,
    REMOVE_CATEGORIES_INPUT,
    DELETE_CATEGORIES,
    ADD_CATEGORY,
    RENAME_CATEGORY,
) = range(12)



def initialize():
    # Any initialization if needed
    logger.info("Initialization complete.")






# Блок 2: Импорт файла и создание категорий
def import_file_and_create_categories(file_content, file_name):
    # Определение количества строк заголовка
    header_rows = 0  # Это значение будет получено из диалога с пользователем в боте

    # Загрузка файла
    if file_name.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(file_content), header=header_rows)
    elif file_name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_content), encoding="utf-8", header=header_rows)
    else:
        raise ValueError(
            "Неподдерживаемый формат файла. Пожалуйста, загрузите файл в формате xlsx или csv."
        )

    # Возвращаем DataFrame и список имен столбцов
    return df, df.columns.tolist()


def get_survey_question():
    # Эта функция будет вызываться из бота
    return "Введите вопрос, на который отвечали участники опроса: "


def get_initial_categories():
    # Эта функция будет вызываться из бота
    categories = []
    # Логика получения начальных категорий будет реализована в боте
    return categories


def suggest_categories(df, open_answer_column, existing_categories, survey_question, max_categories=25):
    answers_series = df[open_answer_column].dropna().astype(str)
    total_answers = len(answers_series)
    sample_size = min(500, total_answers)
    sample_answers = answers_series.sample(n=sample_size, random_state=42).tolist()

    existing_categories_text = (
        ", ".join(existing_categories)
        if existing_categories
        else "Существующие категории отсутствуют."
    )
    sample_answers_text = "\n".join(sample_answers)

    prompt = f"""
<цель>
Вы опытный маркет рисёчер с опытом анализа открытых ответов в опросах. Вам поручено проанализировать открытые ответы в опроснике и предложить категории для классификации.
</цель>

<вопрос>
Вопрос опроса: {survey_question}
</вопрос>

<задача>
Проанализируйте следующие ответы на открытый вопрос и предложите категории для их классификации, опираясь на примеры категорий, представленные ниже, и учитывая контекст вопроса.
{existing_categories_text}
Ответы:
{sample_answers_text}
</задача>

<примеры>
Внимательно прочитайте вопросы и примеры категорий из других исследований, чтобы понять, на что могут быть похожи категории:
## пример1
Вопрос: Что вам нравится в Яндексе?
Ответы:
Привычный / Родной / Давно пользуюсь / Постоянно пользуюсь,
Надежный,
Удобный,
Быстрый,
Простой / понятный,
Находит то что нужно / точный,
Найдет все,
Многофункциональный / Много сервисов / Все в одном месте,
Российский / Отечественный,
Лучше подходит для поиска в России / рядом со мной,
Понимает / учитывает мои интересы,
Доверяю ему / Достоверный,
Красивый / Стильный / Нравится интерфейс,
Пользуюсь другими сервисами бренда,
Современный / Прогрессивный / Идет в ногу со временем,
Лучший / лучше других,
Популярный / На слуху,
Лучше ищет по зарубежным сайтам,
Мало рекламы,
Ненавязчивый (не навязывает контент / сервисы),
Минималистичный / лаконичный (не пестрый / не кричащий / не отвлекает),
Подходит в качестве альтернативного поиска,
Предустановлен на смартфонах / в браузере,
Голосовой помощник / Алиса,
Нейросети / ИИ,
Переводчик,
Безопасный,
Голосовой поиск,
Поиск по картинкам,
Поиск по картинке / фото,
Подходит для некоторых задач,
Нравится / хороший поисковик / все ок / почти нет недостатков,
Не пользуюсь / пользуюсь редко / непривычен,
Есть недочеты / Другие лучше,
Перегруженный / Много лишнего / Пестрый,
Неудобный интерфейс,
Много рекламы,
Плохо ищет,
Плохой дизайн,
Не вызывает доверия,
Навязчиво себя рекламирует,
Следит за пользователями,
Прогосударственный,
Цензура,
Монополист,
Медленный / тормозит,
Хуже подходит для поиска в России / рядом со мной,
Плохой картиночный поиск,
Иностранный продукт,
Сервисы компании без разрешения устанавливаются на компьютер,
Непопулярен,
Сбои / ошибки,
Недовольство другими сервисами бренда,
Не нравится / есть недостатки,
Нет ответа,
Принципиально не рекомендую


## пример2
Вопрос: Почему вы не разрешаете использовать гаджеты при подготовке домашнего задания?
Ответы:
Ребенок должен уметь думать / решать сам,
Чтобы не списывал / не использовал ГДЗ,
Нужно уметь пользоваться учебником и самому искать информацию,
Ребенок не развивается / тупеет (проблемы с логикой, развитием мышления и памяти, фантазией и т.п.),
Ребенок привыкает лениться / слишком расслабляется,
Отвлекают от учебы,
Портят зрение,
Рано использовать гаджеты,
Нет необходимости / все есть в учебниках - можно найти там

## пример3
Вопрос: Опишите, пожалуйста, свои впечатления. Что именно вам понравилось или не понравилось?
Ответы:
Понравилась идея (без уточнения),
Понятное / доступное описание,
Полезно / Удобно,
Новинка / Интересно / Современно,
Помощь от ИИ / Алисы / технологичность,
Помощник / помощь в обучении,
Помогает ребенку - помогает разобраться с заданием и обьясняет тему или решение и логику,
Помогает ребенку - ищет ответы на вопросы/дает ответы на вопросы/можно спросить,
Помощник с ДЗ - выучить уроки, сделать ДЗ,
Помощник - дает подсказки,
Упрощает жизнь родителей - экономит время / обьясняет за них / заменяет ребенку родителей,
Дополнительные занятия - возможность развития и обучения / расширение кругозора ребенка,
Самостоятельная работа ребенка (сам занимается и разбирается в материале / делет задание без привлечения других),
Аналог репетитора  - заменяет живого человека,
Индивидуальный подход к ребенку / учитывает особенности ребенка,
Контроль заданий и анализ ответов ребенка / проверка правильности решения заданий,
Возможность давать материал для закрепления/ примеры похожих заданий,
Пошаговое выполнение заданий,
Актуальность идеи,
Скорость выполнения задания/ Быстро сделать ДЗ/ Быстро найти ответ,
Вызывет интерес у детей будет интересно заниматься/ Повысит мотивацию,
Отсутствие живого общения / Человека нельзя заменить / Отсутствие заботы со стороны родителей,
Гаджеты / телефон / интернет отвлекают  и негативно сказываются на обучении,
Недоверие инновациям и технологиям (негативный отзыв от нейросетях в т.ч.),
Непонятное описание / недостаточно информации/необходимо пробовать/надо смотреть как будет работать,
Неактуально для нас,
Ребенок перестает думать / просто списывает (не развивается / не учится / не думает),
Делает ДЗ за ребенка/ Покажет готовое решение и ответ
</примеры>


<инструкция>
Предложите до {max_categories} дополнительных категорий, которые охватывают основные темы, представленные в ответах, учитывая контекст вопроса опроса. Они должны быть похожи на категории из примеров или даже прямо их повторять, если тематика вопросов подходит.
Следуйте этим рекомендациям:
1. Убедитесь, что предлагаемые категории отличаются от примеров.
2. Категории должны быть достаточно широкими, чтобы сгруппировать похожие ответы, но достаточно конкретными, чтобы быть значимыми.
3. Категории должны максимально полно покрывать имеющиеся ответы и не повторять друг друга.
4. Категорий может быть несколько, но для коротких односложных ответов нужно использовать только одну, наиболее подходящую категорию.
5. Используйте русский язык для названий категорий, за исключением международных брендов, которые должны использовать свои официальные английские названия.
6. Сосредоточьтесь на повторяющихся темах, ключевых понятиях и примечательных упоминаниях в выборочных ответах.
7. Учитывайте как позитивные, так и негативные настроения, если они присутствуют в ответах.
8. Если применимо, включите категории, связанные с характеристиками продукта, опытом клиентов или конкретными случаями использования, упомянутыми в ответах.
Ваш результат должен представлять собой список предложенных категорий, разделенных запятыми, без нумерации и дополнительных пояснений. Внутри категории ни в коем случае не должно быть запятых. Не включайте в список существующие категории.
Верните только список категорий, разделенных запятыми.
</инструкция>
"""

    try:
        response = openai.ChatCompletion.create(
            model=CATEGORIES_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Вы опытный маркет рисёчер с опытом анализа открытых ответов в опросах.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.7,
            n=1,
        )
        categories_output = response.choices[0].message.content.strip()
        suggested_categories = [
            cat.strip()
            for cat in categories_output.split(",")
            if cat.strip() and cat.strip() not in existing_categories
        ]

        # Всегда добавляем "Другое" и "Нерелевантный ответ", если их нет
        if "Другое" not in suggested_categories and "Другое" not in existing_categories:
            suggested_categories.append("Другое")
        if "Нерелевантный ответ" not in suggested_categories and "Нерелевантный ответ" not in existing_categories:
            suggested_categories.append("Нерелевантный ответ")

        return suggested_categories
    except Exception as e:
        logger.error(f"Ошибка при получении предложений категорий: {e}")
        return []

def create_categories(
    df, open_answer_column, survey_question, initial_categories=None, max_categories=25
):
    categories = initial_categories or []

    if categories:
        # Если пользователь ввёл свои категории, используем только их
        return categories, survey_question
    else:
        # Если категорий нет, генерируем их
        max_attempts = 3
        for attempt in range(max_attempts):
            suggested_categories = suggest_categories(
                df, open_answer_column, categories, survey_question, max_categories
            )
            if suggested_categories:
                return suggested_categories, survey_question

        # Если не удалось сгенерировать категории, возвращаем пустой список
        return categories, survey_question



# Блок 3: Анализ
import time

def categorize_answers(df, open_answer_column, categories):
    df["Все категории"] = ""
    df["Обоснование"] = ""
    df["Коды"] = ""
    df["Оценка"] = 0

    # Ensure "Другое" and "Нерелевантный ответ" are in the categories
    if "Другое" not in categories:
        categories.append("Другое")
    if "Нерелевантный ответ" not in categories:
        categories.append("Нерелевантный ответ")

    category_to_code = {category: idx + 1 for idx, category in enumerate(categories)}
    code_to_category = {idx: category for category, idx in category_to_code.items()}

    def create_messages(answer, categories):
        categories_list = ", ".join(categories)
        system_message = "Вы опытный аналитик в области Market Research."
        user_message = f"""
Ваша задача: на основании данного открытого ответа определить, к каким категориям он относится.

Категории: {categories_list}

Ответ: "{answer}"

Проанализируйте ответ и укажите наиболее подходящие категории из предоставленного списка. Если ответ бессодержательный или пустой ("nan"), отнесите его к категории "Нерелевантный ответ". Если ответ содержательный, но не соответствует ни одной предложенной категории, отнесите его к категории "Другое". Убедитесь, что среди предложенных категорий нет похожих на "Другое", например, "Не знаю", и если есть, используйте их. Не используйте категории "Другое" и "Нерелевантный ответ" вместе с другими категориями. Не используйте категории "Другое" и "Нерелевантный ответ" вовсе без необходимости, особенно если есть более подходящие категории. Обоснуйте свой выбор.

Верните результат в формате:

Обоснование: [краткое обоснование выбора категорий]
Категории: [список категорий через запятую]
"""
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def process_answer(args):
        idx, row = args
        answer = row[open_answer_column]
        
        # Check for empty or nonsensical answers
        if pd.isna(answer) or not str(answer).strip() or str(answer).lower() in ['nan', 'none', 'null']:
            return idx, "Нерелевантный ответ", "Ответ пустой или бессмысленный.", str(category_to_code["Нерелевантный ответ"])
        
        messages = create_messages(answer, categories)

        try:
            response = openai.ChatCompletion.create(
                model=ANALYSIS_MODEL_NAME,
                messages=messages,
                max_tokens=150,
                temperature=0,
                n=1,
                stop=None,
            )
            result = response.choices[0].message.content.strip()

            categories_str = ""
            reasoning = ""
            lines = result.split("\n")
            for line in lines:
                if line.startswith("Обоснование:"):
                    reasoning = line.replace("Обоснование:", "").strip()
                elif line.startswith("Категории:"):
                    categories_str = line.replace("Категории:", "").strip()

            # Extract categories from the "Категории" field
            assigned_categories = set()
            for cat in categories_str.split(","):
                cat = cat.strip()
                if cat in categories and cat not in ["Другое", "Нерелевантный ответ"]:
                    assigned_categories.add(cat)
            
            # If no categories were assigned, check the reasoning for negations
            if not assigned_categories:
                negation_words = ["не", "нет", "не соответствует", "не подходит", "не относится"]
                for category in categories:
                    if category in ["Другое", "Нерелевантный ответ"]:
                        continue
                    category_mentioned = category.lower() in reasoning.lower()
                    negated = any(neg in reasoning.lower().split() for neg in negation_words)
                    if category_mentioned and not negated:
                        assigned_categories.add(category)
            
            # If still no categories, use "Другое"
            if not assigned_categories:
                assigned_categories = {"Другое"}

            all_categories_str = ", ".join(sorted(assigned_categories))
            codes = [str(category_to_code[cat]) for cat in assigned_categories]
            codes_str = ", ".join(sorted(codes))

            return idx, all_categories_str, reasoning, codes_str

        except Exception as e:
            logger.error(f"Ошибка при обработке ответа: {e}")
            return idx, "Ошибка", "Ошибка при обработке ответа.", ""

    results = []
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for idx, row in df.iterrows():
            futures.append(executor.submit(process_answer, (idx, row)))

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Категоризация ответов",
        ):
            idx, categories_result, reasoning_result, codes_result = future.result()
            df.at[idx, "Все категории"] = categories_result
            df.at[idx, "Обоснование"] = reasoning_result
            df.at[idx, "Коды"] = codes_result

    return df, category_to_code, code_to_category


def test_categorizations(
    df, open_answer_column, categories, category_to_code, code_to_category, max_iterations=3
):
    def create_messages(answer, categories):
        categories_list = ", ".join(categories)
        system_message = "Вы опытный аналитик в области Market Research."
        user_message = f"""
Ваша задача: на основании данного открытого ответа определить, к каким категориям он относится.

Категории: {categories_list}

Ответ: "{answer}"

Проанализируйте ответ и укажите наиболее подходящие категории из предоставленного списка. Если ответ бессодержательный или пустой ("nan"), отнесите его к категории "Нерелевантный ответ". Если ответ содержательный, но не соответствует ни одной предложенной категории, отнесите его к категории "Другое". Убедитесь, что среди предложенных категорий нет похожих на "Другое", например, "Не знаю", и если есть, используйте их. Не используйте категории "Другое" и "Нерелевантный ответ" вместе с другими категориями. Не используйте категории "Другое" и "Нерелевантный ответ" вовсе без необходимости, особенно если есть более подходящие категории. Обоснуйте свой выбор.
Верните результат в формате:

Обоснование: [краткое обоснование выбора категорий]
Категории: [список категорий через запятую]
"""
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    for iteration in range(max_iterations):
        def evaluate_row(row):
            if "Ошибка" in row["Все категории"] or "Ошибка" in row["Обоснование"]:
                return 0
            elif row["Все категории"] == "" or row["Обоснование"] == "":
                return 1
            else:
                return 2

        df["Оценка"] = df.apply(evaluate_row, axis=1)

        if df["Оценка"].min() >= 2:
            logger.info("Все ответы имеют максимальную оценку.")
            break
        else:
            logger.info(
                f"Итерация {iteration+1}: Повторная категоризация ответов с низкой оценкой."
            )

            df_to_retry = df[df["Оценка"] < 2].copy()

            def retry_process_answer(args):
                idx, row = args
                answer = row[open_answer_column]
                messages = create_messages(answer, categories)

                try:
                    response = openai.ChatCompletion.create(
                        model=ANALYSIS_MODEL_NAME,
                        messages=messages,
                        max_tokens=150,
                        temperature=0,
                        n=1,
                        stop=None,
                    )
                    result = response.choices[0].message.content.strip()

                    categories_str = ""
                    reasoning = ""
                    lines = result.split("\n")
                    for line in lines:
                        if line.startswith("Обоснование:"):
                            reasoning = line.replace("Обоснование:", "").strip()
                        elif line.startswith("Категории:"):
                            categories_str = line.replace("Категории:", "").strip()

                    assigned_categories = [
                        cat.strip()
                        for cat in categories_str.split(",")
                        if cat.strip()
                    ]
                    assigned_categories = [
                        cat if cat in categories else "Другое"
                        for cat in assigned_categories
                    ]
                    all_categories_str = ", ".join(assigned_categories)
                    codes = [
                        str(category_to_code.get(cat, category_to_code["Другое"]))
                        for cat in assigned_categories
                    ]
                    codes_str = ", ".join(codes)

                    return idx, all_categories_str, reasoning, codes_str

                except Exception as e:
                    logger.error(f"Ошибка при повторной обработке ответа: {e}")
                    return idx, "Ошибка", "Ошибка при обработке ответа.", ""

            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                for idx, row in df_to_retry.iterrows():
                    futures.append(executor.submit(retry_process_answer, (idx, row)))

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"Повторная категоризация, итерация {iteration+1}",
                ):
                    idx, categories_result, reasoning_result, codes_result = future.result()
                    df.at[idx, "Все категории"] = categories_result
                    df.at[idx, "Обоснование"] = reasoning_result
                    df.at[idx, "Коды"] = codes_result

    return df


def analyze_category_usage(df, categories, rare_threshold=0.3, max_rare_categories=6):
    all_categories = (
        df["Все категории"]
        .dropna()
        .apply(lambda x: [cat.strip() for cat in x.split(",")])
    )
    all_categories = all_categories.explode()
    category_counts = all_categories.value_counts()

    total_counts = category_counts.sum()
    average_count = total_counts / len(category_counts)
    
    # Calculate the threshold for rare categories
    threshold = max(average_count * rare_threshold, 2)  # At least 2 occurrences

    # Identify rare categories
    rare_categories = category_counts[category_counts < threshold]
    
    # Sort rare categories by frequency and limit to max_rare_categories
    rare_categories = rare_categories.sort_values().head(max_rare_categories)

    common_categories = category_counts[category_counts >= threshold]

    logger.info(f"Number of rare categories: {len(rare_categories)}")
    logger.info(f"Number of common categories: {len(common_categories)}")
    
    # Add this line to log the actual rare categories
    logger.info(f"Rare categories: {rare_categories.to_dict()}")

    return categories, rare_categories, common_categories






def create_category_columns(df, categories, category_to_code):
    # Ensure "Другое" and "Нерелевантный ответ" are in the categories
    if "Другое" not in categories:
        categories.append("Другое")
        category_to_code["Другое"] = max(category_to_code.values()) + 1
    if "Нерелевантный ответ" not in categories:
        categories.append("Нерелевантный ответ")
        category_to_code["Нерелевантный ответ"] = max(category_to_code.values()) + 1

    for category, code in category_to_code.items():
        column_name = f"{code}. {category}"
        df[column_name] = df["Коды"].apply(
            lambda x: category if str(code) in x.split(", ") else ""
        )
    return df






import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import io

def save_results(df, code_to_category):
  # Remove the "Оценка" column if it exists
    if "Оценка" in df.columns:
        df = df.drop(columns=["Оценка"])
        logger.info('"Оценка" column removed from the DataFrame.')

    # Add separate category columns using category names
    categories = list(code_to_category.values())
    df = create_category_columns(df, categories, code_to_category)
    logger.info("Separate category columns added to the DataFrame.")

    # Calculate frequency of each category
    all_categories = df["Все категории"].dropna().apply(lambda x: [cat.strip() for cat in x.split(",")])
    all_categories = all_categories.explode()
    category_counts = all_categories.value_counts()

    xlsx_buffer = io.BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
        # Add instruction sheet
        intro_text = """
Это внутренний ИИ-категоризатор открытых ответов, сделанный в UX лаборатории Яндекса.
Проанализированные ответы находятся на следующей вкладке.
По вопросам можно писать Лёше Шипулину shipaleks@ (t.me/bdndjcmf)
"""
        intro_sentences = [sentence.strip() for sentence in intro_text.strip().split('\n') if sentence.strip()]
        intro_df = pd.DataFrame(intro_sentences, columns=["Инструкция"])
        intro_df.to_excel(writer, index=False, sheet_name="Инструкция")
        logger.info("Instruction sheet added to Excel.")

        # Add results sheet
        df.to_excel(writer, index=False, sheet_name="Результаты")
        logger.info("Results sheet added to Excel.")

        # Add codes, categories, and frequencies sheet
        code_category_freq = pd.DataFrame({
            'Код': code_to_category.keys(),
            'Категория': code_to_category.values(),
            'Частота': [category_counts.get(cat, 0) for cat in code_to_category.values()]
        })
        code_category_freq = code_category_freq.sort_values('Код')
        code_category_freq.to_excel(writer, index=False, sheet_name="Коды и категории")
        logger.info("Codes, Categories, and Frequencies sheet added to Excel.")

        # Format instruction sheet
        workbook = writer.book
        instruction_sheet = writer.sheets["Инструкция"]
        instruction_format = workbook.add_format({
            'text_wrap': True, 'valign': 'top', 'font_size': 12,
            'font_name': 'Calibri', 'font_color': '#000000'
        })
        instruction_sheet.set_column('A:A', 100, instruction_format)
        for row_num in range(len(intro_sentences)):
            instruction_sheet.set_row(row_num, 30)

    # Построение графика с использованием seaborn
    all_categories = (
        df["Все категории"]
        .dropna()
        .apply(lambda x: [cat.strip() for cat in x.split(",")])
    )
    all_categories = all_categories.explode()
    category_counts = all_categories.value_counts()

    # Настройка стиля и цветов с добавлением серой сетки
    sns.set(style="whitegrid", rc={
        "grid.color": "#F0F1F4",          # Цвет сетки
        "grid.linestyle": "--",        # Стиль линий сетки
        "grid.linewidth": 1.5           # Толщина линий сетки
    })

    plt.figure(figsize=(12, 6), facecolor='#F0F1F4')  # Устанавливаем цвет фона

    # Создаём горизонтальную гистограмму с заданным цветом столбцов
    ax = sns.barplot(x=category_counts.values, y=category_counts.index, color='#F8604A')

    # Настройка цветов текста
    ax.set_title("Распределение категорий", color='black', fontsize=16, fontweight='bold')
    ax.set_xlabel("Количество", color='black', fontsize=12)
    ax.set_ylabel("Категории", color='black', fontsize=12)

    # Настройка цветов осей и подписей
    ax.tick_params(colors='black', labelsize=10)  # Цвет текста на осях

    # Настройка цвета линий осей
    ax.spines['bottom'].set_color('#F0F1F4')  # Цвет линий осей
    ax.spines['left'].set_color('#F0F1F4')
    ax.spines['top'].set_color('#F0F1F4')
    ax.spines['right'].set_color('#F0F1F4')

    # Добавление меток на столбцы
    for p in ax.patches:
        ax.annotate(format(p.get_width(), '.0f'),
                    (p.get_width() + 1, p.get_y() + p.get_height() / 2),
                    ha='left', va='center', color='black', fontsize=10)

    plt.tight_layout()

    # Сохранение графика в буфер
    png_buffer = io.BytesIO()
    plt.savefig(png_buffer, format="png", facecolor=plt.gcf().get_facecolor())
    plt.close()
    logger.info("Distribution chart generated and saved to buffer.")

    return xlsx_buffer.getvalue(), png_buffer.getvalue()



def column_letter_to_index(letter: str) -> int:
    """
    Converts an Excel-style column letter (e.g., 'A', 'B', ..., 'AA', etc.) to a zero-based index.
    
    Args:
        letter (str): The column letter.
    
    Returns:
        int: The zero-based column index.
    
    Raises:
        ValueError: If the input is not a valid Excel column letter.
    """
    letter = letter.strip().upper()
    if not letter.isalpha():
        raise ValueError("Неверный формат столбца. Пожалуйста, введите букву столбца (например, A, B, C).")
    
    index = 0
    for char in letter:
        if 'A' <= char <= 'Z':
            index = index * 26 + (ord(char) - ord('A') + 1)
        else:
            raise ValueError("Неверный формат столбца. Пожалуйста, используйте только буквы A-Z.")
    return index - 1  # Zero-based index





# Основная функция для обработки опроса

def process_survey_data(
    file_content,
    file_name,
    survey_question,
    open_answer_column,
    initial_categories=None,
):
    logger.info(
        f"Received arguments: file_name={file_name}, survey_question={survey_question}, open_answer_column={open_answer_column}"
    )
    initialize()
    df, columns = import_file_and_create_categories(file_content, file_name)
    open_answer_column = columns[open_answer_column]
    categories, _ = create_categories(
        df, open_answer_column, survey_question, initial_categories
    )
    df, category_to_code, code_to_category = categorize_answers(
        df, open_answer_column, categories
    )
    df = test_categorizations(
        df, open_answer_column, categories, category_to_code, code_to_category
    )
    categories, rare_categories, common_categories = analyze_category_usage(
        df, categories
    )

    # **Add this line to create category columns**
    df = create_category_columns(df, categories, category_to_code)

    # **Now return the modified DataFrame with the new columns**
    return df, category_to_code, code_to_category, rare_categories

def process_survey_data_sync(file_content, file_name, survey_question, open_answer_column, initial_categories=None):
    df, columns = import_file_and_create_categories(file_content, file_name)
    open_answer_column = columns[open_answer_column]
    categories, _ = create_categories(
        df, open_answer_column, survey_question, initial_categories
    )
    df, category_to_code, code_to_category = categorize_answers(
        df, open_answer_column, categories
    )
    df = test_categorizations(
        df, open_answer_column, categories, category_to_code, code_to_category
    )
    categories, rare_categories, common_categories = analyze_category_usage(df, categories)
    
    df = create_category_columns(df, categories, category_to_code)

    return df, category_to_code, code_to_category, rare_categories


def get_user_data(user_id):
    user_ref = ref.child('users').child(str(user_id))
    user_data = user_ref.get()
    if user_data is None:
        user_data = {'free_answers': 0}
        user_ref.set(user_data)
    return user_data

def update_user_free_answers(user_id, new_free_answers):
    user_ref = ref.child('users').child(str(user_id))
    user_ref.update({'free_answers': new_free_answers})

def check_free_answers_limit(user_id):
    user_data = get_user_data(user_id)
    return user_data['free_answers'] < 2000

async def send_payment_request(update: Update, context: ContextTypes.DEFAULT_TYPE, paid_answers):
    chat_id = update.effective_chat.id
    title = "Оплата за анализ дополнительных ответов"
    description = f"Анализ {paid_answers} дополнительных ответов"
    payload = f"paid_answers_{paid_answers}"
    currency = "XTR"
    # Assuming 1 XTR per answer; adjust according to your pricing
    amount_per_answer = 1  # Adjust this value based on your pricing strategy
    price = paid_answers * amount_per_answer
    prices = [LabeledPrice("Анализ ответов", int(price))]

    # For payments in Telegram Stars, provider_token should be an empty string
    provider_token = ""

    try:
        await context.bot.send_invoice(
            chat_id=chat_id,
            title=title,
            description=description,
            payload=payload,
            provider_token=provider_token,
            currency=currency,
            prices=prices,
            # The following parameters are ignored for Telegram Stars payments
            need_name=False,
            need_phone_number=False,
            need_email=False,
            need_shipping_address=False,
            is_flexible=False,
            # Optional: Remove or comment out if not needed
            # reply_markup=your_custom_inline_keyboard,
        )
    except Exception as e:
        logger.error(f"Ошибка при отправке запроса на оплату: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            text="Произошла ошибка при формировании запроса на оплату. Пожалуйста, попробуйте позже."
        )

async def send_results(update: Update, context: ContextTypes.DEFAULT_TYPE, xlsx_content, png_content):
    chat_id = update.effective_chat.id
    
    # Отправка Excel файла
    await context.bot.send_document(
        chat_id=chat_id,
        document=io.BytesIO(xlsx_content),
        filename="results.xlsx",
        caption="Результаты анализа (Excel файл)"
    )
    
    # Отправка изображения с графиком
    await context.bot.send_photo(
        chat_id=chat_id,
        photo=io.BytesIO(png_content),
        caption="Распределение категорий"
    )
    
    await context.bot.send_message(
        chat_id=chat_id,
        text="Анализ завершен. Результаты отправлены."
    )

async def successful_payment_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    payment = update.message.successful_payment
    paid_answers = int(payment.invoice_payload.split('_')[2])
    
    # Retrieve the full survey data from user_data
    df, category_to_code, code_to_category, _ = context.user_data['survey_data']
    xlsx_content, png_content = save_results(df, code_to_category)
    await send_results(update, context, xlsx_content, png_content)

    await update.message.reply_text(f"Спасибо за оплату! Анализ {paid_answers} ответов выполнен.")


async def pre_checkout_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.pre_checkout_query
    await query.answer(ok=True)


async def check_free_answers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = get_user_data(user_id)
    remaining_free_answers = max(0, 2000 - user_data['free_answers'])
    await update.message.reply_text(f"У вас осталось {remaining_free_answers} бесплатных ответов.")



# Функции из bot.py

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Проверка, есть ли существующие данные, что указывает на перезапуск
    if context.user_data:
        await update.message.reply_text(
            "Процесс перезапущен. Все предыдущие данные очищены."
        )

    # Очистка любых существующих данных пользователя
    context.user_data.clear()

    # Установка значения по умолчанию для max_categories
    context.user_data["max_categories"] = 25  # Значение по умолчанию


    # Асинхронное получение общего количества проанализированных ответов из Firebase
    try:
        loop = asyncio.get_event_loop()
        total_answers = await loop.run_in_executor(None, get_total_answers)
    except Exception as e:
        logger.error(f"Не удалось получить total_answers: {e}", exc_info=True)
        total_answers = 0  # По умолчанию 0, если произошла ошибка

    # Логирование события запуска
    logger.info(f"Пользователь {update.effective_user.id} запустил бота. Общее количество проанализированных ответов: {total_answers}")

    # Расчет экономии
    agency_cost_per_answer = 5  # рублей
    agency_time_per_answer_min = 0.5  # 30 секунд
    bot_cost_per_answer = 2  # рубля
    bot_time_per_1000_answers_min = 5  # минут
    bot_time_per_answer_min = bot_time_per_1000_answers_min / 1000  # 0.005 минут

    # Общий расчет экономии
    total_money_saved = (agency_cost_per_answer - bot_cost_per_answer) * total_answers
    total_time_saved_min = (agency_time_per_answer_min - bot_time_per_answer_min) * total_answers

    # Форматирование экономии для удобочитаемости
    total_money_saved_str = f"{total_money_saved:,.2f} ₽"
    if total_time_saved_min >= 60:
        hours = total_time_saved_min // 60
        minutes = total_time_saved_min % 60
        total_time_saved_str = f"{int(hours)} часов {int(minutes)} минут"
    else:
        total_time_saved_str = f"{total_time_saved_min:.2f} минут"

    # Формирование нового приветственного сообщения
    greeting_message = (
        f"Привет! 🤖 Я ваш умный помощник по анализу опросов.\n\n"
        f"🧠 *Что я умею:*\n"
        f"• Автоматически анализирую открытые ответы из ваших опросов с помощью AI\n"
        f"• Категоризирую ответы для удобства обработки и анализа\n\n"
        f"⚡ *Почему я лучше традиционных методов:*\n"
        f"• *Скорость:* Обрабатываю 1000 ответов всего за 5 минут — вместо целого рабочего дня у агентства\n"
        f"• *Экономия:* Стоимость 1⭐️ (около 2 ₽) за ответ — против 5 ₽ в агентствах\n\n"
        f"💰 *Выгода:*\n"
        f"За всё время использования бота клиенты сэкономили *{total_money_saved_str}* и *{total_time_saved_str}* по сравнению с обработкой вручную.\n\n"
        f"🎁 *Специальное предложение:*\n"
        f"Попробуйте бесплатно! Проанализируйте *2000 ответов бесплатно* и оцените качество сервиса без риска.\n\n"
        f"🚀 *Готовы начать?*\n"
        f"*Загрузите ваш файл с данными (.xlsx или .csv).*\n"
        f"В файле должна быть только одна вкладка."
            )

    await update.message.reply_text(greeting_message, parse_mode='Markdown')
    return UPLOAD





# Обработчик загрузки файла
async def file_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    file = await update.message.document.get_file()
    file_content = await file.download_as_bytearray()
    context.user_data["file_content"] = file_content
    context.user_data["file_name"] = update.message.document.file_name

    # Import file and store columns
    try:
        df, columns = import_file_and_create_categories(file_content, update.message.document.file_name)
        context.user_data["columns"] = columns
        await update.message.reply_text(
            "Файл успешно загружен. Теперь введите вопрос, на который отвечали участники опроса."
        )
        return QUESTION
    except Exception as e:
        logger.error(f"Error importing file: {e}", exc_info=True)
        await update.message.reply_text(
            "Произошла ошибка при загрузке файла. Пожалуйста, убедитесь, что файл в формате .xlsx или .csv и попробуйте снова."
        )
        return ConversationHandler.END



# Обработчик ввода вопроса
async def question_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["survey_question"] = update.message.text

    await update.message.reply_text(
        "Вопрос сохранен. Теперь введите букву столбца с ответами (например, A, B, C)."
    )
    return COLUMN


# Обработчик выбора столбца
async def column_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_input = update.message.text.strip().upper()
    try:
        column_index = column_letter_to_index(user_input)
        columns = context.user_data.get("columns", [])
        
        if column_index < 0 or column_index >= len(columns):
            raise ValueError("Указанный столбец выходит за пределы загруженного файла.")
        
        context.user_data["open_answer_column"] = column_index

        keyboard = [
            [
                InlineKeyboardButton("Да", callback_data="yes"),
                InlineKeyboardButton("Нет", callback_data="no"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        selected_column_name = columns[column_index]
        await update.message.reply_text(
            f"Вы выбрали столбец {user_input} ({selected_column_name}). У вас есть готовые категории для анализа?", reply_markup=reply_markup
        )

        return CATEGORIES
    except ValueError as ve:
        await update.message.reply_text(f"Ошибка: {ve}")
        return COLUMN
    except Exception as e:
        logger.error(f"Unexpected error in column_handler: {e}", exc_info=True)
        await update.message.reply_text("Произошла ошибка при обработке вашего ввода. Пожалуйста, попробуйте еще раз.")
        return COLUMN


# Обработчик выбора категорий
async def categories_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    if query.data == "yes":
        await query.edit_message_text("Пожалуйста, введите категории через запятую или через новую строку (в столбик)")
        return CATEGORIES
    else:
        context.user_data["categories"] = []
        return await suggest_categories_handler(update, context)


async def edit_categories_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    query = update.callback_query
    await query.answer()

    if query.data == "delete_categories":
        # Отправляем список категорий с номерами для удаления
        categories = context.user_data.get("categories", context.user_data.get("suggested_categories", []))
        categories_text = "\n".join(
            [f"{i+1}. {cat}" for i, cat in enumerate(categories)]
        )
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Текущие категории:\n\n{categories_text}\n\nВведите номера категорий, которые вы хотите удалить, через запятую.",
        )
        return DELETE_CATEGORIES
    elif query.data == "add_category":
        # Отправляем список текущих категорий перед добавлением новых
        categories = context.user_data.get("categories", context.user_data.get("suggested_categories", []))
        categories_text = "\n".join(
            [f"{i+1}. {cat}" for i, cat in enumerate(categories)]
        )
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Текущие категории:\n\n{categories_text}\n\nВведите названия категорий, которые вы хотите добавить, через запятую.",
        )
        return ADD_CATEGORY
    elif query.data == "rename_category":
        # Отправляем список категорий с номерами для переименования
        categories = context.user_data.get("categories", context.user_data.get("suggested_categories", []))
        categories_text = "\n".join(
            [f"{i+1}. {cat}" for i, cat in enumerate(categories)]
        )
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Текущие категории:\n\n{categories_text}\n\nВведите номер категории, которую вы хотите переименовать, и новое название через двоеточие. Например: 3: Новое название категории",
        )
        return RENAME_CATEGORY
    elif query.data == "finish_editing":
        # Заканчиваем редактирование и предлагаем начать анализ
        categories = context.user_data.get("categories", context.user_data.get("suggested_categories", []))
        categories_text = "\n".join(
            [f"{i+1}. {cat}" for i, cat in enumerate(categories)]
        )
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Текущие категории:\n\n{categories_text}\n\nНачать анализ?",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton("Да", callback_data="start_analysis"),
                        InlineKeyboardButton("Нет", callback_data="cancel"),
                    ]
                ]
            ),
        )
        return CHOOSE_CATEGORIES
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Неизвестный выбор. Пожалуйста, попробуйте еще раз.",
        )
        return EDIT_CATEGORIES


# Добавьте эти функции в ваш файл с кодом бота

async def payment_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    payment_info = (
    "⭐️ *Что такое Telegram Stars?*\n\n"
    "Telegram Stars - это виртуальная валюта внутри Telegram для оплаты цифровых услуг, "
    "включая функции нашего бота.\n\n"
    "🤔 *Почему мы используем Telegram Stars?*\n\n"
    "Недавно правила App Store и Google Play изменились, запретив приложениям принимать "
    "прямые платежи за цифровые услуги. Использование Stars - это способ продолжить "
    "предоставлять вам наши услуги в соответствии с новыми правилами.\n\n"
    "🧾 *Как купить Telegram Stars?*\n\n"
    "1. Через приложение Telegram (немного дороже):\n"
    "   - iOS: Настройки → Мои звезды → Купить больше звезд\n"
    "   - Android: Настройки → Мои звезды\n"
    "2. Через бота [@PremiumBot](https://t.me/PremiumBot) (обычно выгоднее)\n\n"
    "💰 *Как сэкономить при покупке Stars?*\n\n"
    "Рекомендуем покупать Stars через [@PremiumBot](https://t.me/PremiumBot) - это заметно выгоднее. "
    "Например, 250 звёзд там стоят 465,99 ₽, а через App Store - 599 ₽.\n\n"
    "Также, покупка большего количества Stars за раз обычно даёт лучшую цену за единицу. "
    "Если вы планируете регулярно пользоваться ботом, возможно, стоит рассмотреть покупку "
    "большего пакета Stars."
)
    
    await update.message.reply_text(payment_info, parse_mode='Markdown', disable_web_page_preview=True)

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_message = (
        "Если у вас возникли вопросы или нужна помощь, пожалуйста, свяжитесь с создателем бота: [@bdndjcmf](https://t.me/bdndjcmf)"
    )
    await update.message.reply_text(help_message, parse_mode='Markdown', disable_web_page_preview=True)




async def delete_categories_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        categories_to_remove = [
            int(x.strip())
            for x in update.message.text.split(",")
            if x.strip().isdigit()
        ]
        categories = context.user_data.get("categories", context.user_data.get("suggested_categories", []))
        context.user_data["categories"] = [
            cat
            for i, cat in enumerate(categories, 1)
            if i not in categories_to_remove
        ]

        categories = context.user_data["categories"]
        categories_text = "\n".join(
            [f"{i+1}. {cat}" for i, cat in enumerate(categories)]
        )

        await update.message.reply_text(
            f"Категории удалены. Текущие категории:\n\n{categories_text}\n\nВыберите следующее действие:",
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("🗑️ Удалить категории", callback_data="delete_categories")],
                    [InlineKeyboardButton("➕ Добавить категорию", callback_data="add_category")],
                    [InlineKeyboardButton("✏️ Изменить название категории", callback_data="rename_category")],
                    [InlineKeyboardButton("🚀 Запустить анализ", callback_data="finish_editing")],
                ]
            ),
        )
        return EDIT_CATEGORIES
    except Exception as e:
        logger.error(f"Ошибка в delete_categories_handler: {e}")
        await update.message.reply_text(
            "Произошла ошибка при удалении категорий. Пожалуйста, попробуйте еще раз."
        )
        return EDIT_CATEGORIES


from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes, ConversationHandler
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Предполагается, что состояние EDIT_CATEGORIES уже определено
# Например:
# EDIT_CATEGORIES = range(1)

async def add_category_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        # Получаем текст сообщения пользователя и удаляем лишние пробелы
        message_text = update.message.text.strip()
        
        if not message_text:
            await update.message.reply_text(
                "❌ Вы не отправили никаких категорий. Пожалуйста, отправьте категории через запятые или переносы строк."
            )
            return EDIT_CATEGORIES

        # Используем регулярное выражение для разделения по запятым и переносам строк
        # Разделители: запятая, перенос строки \n, \r\n
        categories = [cat.strip() for cat in re.split(r',|\r?\n', message_text) if cat.strip()]

        if not categories:
            await update.message.reply_text(
                "❌ Не удалось распознать категории. Пожалуйста, убедитесь, что вы правильно разделили их запятыми или переносами строк."
            )
            return EDIT_CATEGORIES

        # Получаем текущие категории пользователя из context.user_data
        current_categories = context.user_data.get("categories", context.user_data.get("suggested_categories", []))

        # Убираем дубликаты и добавляем только новые категории
        new_unique_categories = []
        for new_cat in categories:
            if new_cat not in current_categories:
                current_categories.append(new_cat)
                new_unique_categories.append(new_cat)

        # Обновляем категории в context.user_data
        context.user_data["categories"] = current_categories

        if not new_unique_categories:
            await update.message.reply_text(
                "✅ Все отправленные категории уже существуют. Текущие категории:\n\n" + "\n".join(
                    [f"{i+1}. {cat}" for i, cat in enumerate(current_categories)]
                ) + "\n\nВыберите следующее действие:",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("🗑️ Удалить категории", callback_data="delete_categories")],
                        [InlineKeyboardButton("➕ Добавить категорию", callback_data="add_category")],
                        [InlineKeyboardButton("✏️ Изменить название категории", callback_data="rename_category")],
                        [InlineKeyboardButton("🚀 Запустить анализ", callback_data="finish_editing")],
                    ]
                ),
            )
            return EDIT_CATEGORIES

        # Формирование текста с новыми добавленными категориями
        categories_text = "\n".join(
            [f"{i+1}. {cat}" for i, cat in enumerate(current_categories)]
        )

        await update.message.reply_text(
            f"✅ Добавлены категории:\n" + "\n".join([f"- {cat}" for cat in new_unique_categories]) +
            f"\n\nТекущие категории:\n\n{categories_text}\n\nВыберите следующее действие:",
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("🗑️ Удалить категории", callback_data="delete_categories")],
                    [InlineKeyboardButton("➕ Добавить категорию", callback_data="add_category")],
                    [InlineKeyboardButton("✏️ Изменить название категории", callback_data="rename_category")],
                    [InlineKeyboardButton("🚀 Запустить анализ", callback_data="finish_editing")],
                ]
            ),
        )
        return EDIT_CATEGORIES

    except Exception as e:
        logger.error(f"Ошибка в add_category_handler: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Произошла ошибка при добавлении категорий. Пожалуйста, попробуйте еще раз."
        )
        return EDIT_CATEGORIES


        # Формирование текста с новыми добавленными категориями
        categories_text = "\n".join(
            [f"{i+1}. {cat}" for i, cat in enumerate(current_categories)]
        )

        await update.message.reply_text(
            f"✅ Добавлены категории:\n" + "\n".join([f"- {cat}" for cat in new_unique_categories]) +
            f"\n\nТекущие категории:\n\n{categories_text}\n\nВыберите следующее действие:",
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("🗑️ Удалить категории", callback_data="delete_categories")],
                    [InlineKeyboardButton("➕ Добавить категорию", callback_data="add_category")],
                    [InlineKeyboardButton("✏️ Изменить название категории", callback_data="rename_category")],
                    [InlineKeyboardButton("🚀 Запустить анализ", callback_data="finish_editing")],
                ]
            ),
        )
        return EDIT_CATEGORIES

    except Exception as e:
        logger.error(f"Ошибка в add_category_handler: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Произошла ошибка при добавлении категорий. Пожалуйста, попробуйте еще раз."
        )
        return EDIT_CATEGORIES




async def rename_category_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        user_input = update.message.text.strip()
        if ":" not in user_input:
            await update.message.reply_text(
                "Пожалуйста, введите данные в формате 'номер: новое название'. Например: 3: Новое название категории"
            )
            return RENAME_CATEGORY

        index_str, new_name = user_input.split(":", 1)
        index = int(index_str.strip())
        new_name = new_name.strip()

        categories = context.user_data.get("categories", context.user_data.get("suggested_categories", []))
        if index < 1 or index > len(categories):
            await update.message.reply_text(
                "Некорректный номер категории. Пожалуйста, попробуйте еще раз."
            )
            return RENAME_CATEGORY

        categories[index - 1] = new_name
        context.user_data["categories"] = categories

        categories_text = "\n".join(
            [f"{i+1}. {cat}" for i, cat in enumerate(categories)]
        )

        await update.message.reply_text(
            f"Категория переименована. Текущие категории:\n\n{categories_text}\n\nВыберите следующее действие:",
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("🗑️ Удалить категории", callback_data="delete_categories")],
                    [InlineKeyboardButton("➕ Добавить категорию", callback_data="add_category")],
                    [InlineKeyboardButton("✏️ Изменить название категории", callback_data="rename_category")],
                    [InlineKeyboardButton("🚀 Запустить анализ", callback_data="finish_editing")],
                ]
            ),
        )
        return EDIT_CATEGORIES
    except Exception as e:
        logger.error(f"Ошибка в rename_category_handler: {e}")
        await update.message.reply_text(
            "Произошла ошибка при переименовании категории. Пожалуйста, попробуйте еще раз."
        )
        return RENAME_CATEGORY




# Обработчик ввода категорий вручную
async def manual_categories_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        # Получаем текст сообщения пользователя и удаляем лишние пробелы
        message_text = update.message.text.strip()
        
        if not message_text:
            await update.message.reply_text(
                "❌ Вы не отправили никаких категорий. Пожалуйста, отправьте категории через запятые или переносы строк."
            )
            return CATEGORIES  # Или EDIT_CATEGORIES в зависимости от логики

        # Используем регулярное выражение для разделения по запятым и переносам строк
        # Разделители: запятая, перенос строки \n, \r\n
        categories = [cat.strip() for cat in re.split(r',|\r?\n', message_text) if cat.strip()]

        if not categories:
            await update.message.reply_text(
                "❌ Не удалось распознать категории. Пожалуйста, убедитесь, что вы правильно разделили их запятыми или переносами строк."
            )
            return CATEGORIES  # Или EDIT_CATEGORIES в зависимости от логики

        # Получаем текущие категории пользователя из context.user_data
        current_categories = context.user_data.get("categories", context.user_data.get("suggested_categories", []))

        # Убираем дубликаты и добавляем только новые категории
        new_unique_categories = []
        for new_cat in categories:
            if new_cat not in current_categories:
                current_categories.append(new_cat)
                new_unique_categories.append(new_cat)

        # Обновляем категории в context.user_data
        context.user_data["categories"] = current_categories

        if not new_unique_categories:
            await update.message.reply_text(
                "✅ Все отправленные категории уже существуют. Текущие категории:\n\n" + "\n".join(
                    [f"{i+1}. {cat}" for i, cat in enumerate(current_categories)]
                ) + "\n\nВыберите следующее действие:",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("🗑️ Удалить категории", callback_data="delete_categories")],
                        [InlineKeyboardButton("➕ Добавить категорию", callback_data="add_category")],
                        [InlineKeyboardButton("✏️ Изменить название категории", callback_data="rename_category")],
                        [InlineKeyboardButton("🚀 Запустить анализ", callback_data="finish_editing")],
                    ]
                ),
            )
            return EDIT_CATEGORIES

        # Формирование текста с новыми добавленными категориями
        categories_text = "\n".join(
            [f"{i+1}. {cat}" for i, cat in enumerate(current_categories)]
        )

        await update.message.reply_text(
            f"✅ Добавлены категории:\n" + "\n".join([f"- {cat}" for cat in new_unique_categories]) +
            f"\n\nТекущие категории:\n\n{categories_text}\n\nВыберите следующее действие:",
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("🗑️ Удалить категории", callback_data="delete_categories")],
                    [InlineKeyboardButton("➕ Добавить категорию", callback_data="add_category")],
                    [InlineKeyboardButton("✏️ Изменить название категории", callback_data="rename_category")],
                    [InlineKeyboardButton("🚀 Запустить анализ", callback_data="finish_editing")],
                ]
            ),
        )
        return EDIT_CATEGORIES

    except Exception as e:
        logger.error(f"Ошибка в manual_categories_handler: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Произошла ошибка при добавлении категорий. Пожалуйста, попробуйте еще раз."
        )
        return EDIT_CATEGORIES



# Обработчик отмены
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Операция отменена.")
    return ConversationHandler.END


async def suggest_categories_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.debug("Entering suggest_categories_handler")
    try:
        chat_id = update.effective_chat.id
        # Импорт файла и создание DataFrame
        df, columns = import_file_and_create_categories(
            context.user_data["file_content"], context.user_data["file_name"]
        )
        # Получение столбца с открытыми ответами
        open_answer_column = columns[context.user_data["open_answer_column"]]
        survey_question = context.user_data["survey_question"]
        
        # Используем пустой список для initial_categories, чтобы форсировать генерацию
        initial_categories = []

        # Получение текущего значения max_categories
        max_categories = context.user_data.get("max_categories", 25)

        # Генерация предложенных категорий
        loop = asyncio.get_event_loop()
        suggested_categories, _ = await loop.run_in_executor(
            None,
            create_categories,
            df,
            open_answer_column,
            survey_question,
            initial_categories,
            max_categories,  # Передаём max_categories
        )

        # Сохранение предложенных категорий в user_data
        context.user_data["categories"] = suggested_categories

        # Подготовка клавиатуры с новыми кнопками
        keyboard = [
            [InlineKeyboardButton("🚀 Запустить анализ", callback_data="use_categories")],
            [InlineKeyboardButton("✏️ Редактировать", callback_data="edit")],
            [InlineKeyboardButton("⬆️ Сгенерировать побольше (до 50 категорий)", callback_data="increase_categories")],
            [InlineKeyboardButton("⬇️ Сгенерировать поменьше (до 10 категорий)", callback_data="decrease_categories")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Добавление нумерации к категориям для отображения
        categories = context.user_data["categories"]
        categories_text = "\n".join(
            [f"{i+1}. {cat}" for i, cat in enumerate(categories)]
        )

        # Отправка сообщения с предложенными категориями и опциями
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Предлагаемые категории:\n\n{categories_text}\n\nЧто вы хотите сделать?",
            reply_markup=reply_markup,
        )

        logger.debug("Sent message with categories and buttons. Returning CHOOSE_CATEGORIES state.")
        return CHOOSE_CATEGORIES
    except Exception as e:
        logger.error(f"Error in suggest_categories_handler: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Произошла ошибка при генерации категорий. Пожалуйста, проверьте, что указали верный столбец (у столбца А номер 0, у B номер 1, и так далее)",
        )
        return ConversationHandler.END






# Обязательно объявите, что вы будете изменять глобальную переменную
async def process_survey_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    await context.bot.send_message(
        chat_id=chat_id,
        text="Начинаю обработку опроса. Это займёт несколько минут...",
    )

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            process_survey_data_sync,
            context.user_data["file_content"],
            context.user_data["file_name"],
            context.user_data["survey_question"],
            context.user_data["open_answer_column"],
            context.user_data.get("categories"),
        )

        df, category_to_code, code_to_category, rare_categories = result
        total_answers = len(df)

        # Сохраняем данные для возможного дальнейшего использования
        context.user_data['survey_data'] = (df, category_to_code, code_to_category, rare_categories)

        # Проверяем наличие редких категорий
        if isinstance(rare_categories, pd.Series) and not rare_categories.empty:
            logger.info(f"Редкие категории обнаружены: {rare_categories.to_dict()}")
            return await handle_rare_categories(update, context, rare_categories)
        elif isinstance(rare_categories, list) and rare_categories:
            logger.info(f"Редкие категории обнаружены: {rare_categories}")
            return await handle_rare_categories(update, context, rare_categories)
        else:
            logger.info("Редкие категории не обнаружены")

        # Если нет редких категорий, продолжаем обработку
        return await process_final_results(update, context)

    except Exception as e:
        logger.error(f"Ошибка при обработке опроса: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text="Произошла ошибка при обработке опроса. Пожалуйста, попробуйте еще раз.",
        )

    return ConversationHandler.END
    
async def handle_rare_categories(update: Update, context: ContextTypes.DEFAULT_TYPE, rare_categories=None) -> int:
    chat_id = update.effective_chat.id
    
    if rare_categories is not None:
        if isinstance(rare_categories, pd.Series):
            rare_categories_with_counts = list(rare_categories.items())
        elif isinstance(rare_categories, dict):
            rare_categories_with_counts = list(rare_categories.items())
        else:
            logger.error(f"Неожиданный тип редких категорий: {type(rare_categories)}")
            await context.bot.send_message(
                chat_id=chat_id,
                text="Произошла ошибка при обработке редких категорий. Пожалуйста, попробуйте еще раз.",
            )
            return ConversationHandler.END
        
        context.user_data["rare_categories_with_counts"] = rare_categories_with_counts
    else:
        rare_categories_with_counts = context.user_data.get("rare_categories_with_counts", [])
    
    if not rare_categories_with_counts:
        logger.info("rare_categories_with_counts is empty in handle_rare_categories")
        return await process_final_results(update, context)

    rare_categories_text = "\n".join(
        [f"{i}. {cat} (упоминаний: {count})" for i, (cat, count) in enumerate(rare_categories_with_counts, start=1)]
    )

    message = (
        f"Следующие категории используются реже остальных:\n\n{rare_categories_text}\n\n"
        "Хотите ли вы удалить некоторые из этих категорий перед финальной обработкой? Это бесплатно."
    )

    keyboard = [
        [InlineKeyboardButton("🗑️ Выбрать, какие удалить", callback_data="choose_to_delete")],
        [InlineKeyboardButton("📊 Оставить все категории", callback_data="keep_all")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await context.bot.send_message(chat_id=chat_id, text=message, reply_markup=reply_markup)
    return RARE_CATEGORIES


async def ask_for_categories_to_remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id if update.effective_chat else update.callback_query.message.chat_id
    rare_categories_with_counts = context.user_data.get("rare_categories_with_counts", [])

    rare_categories_text = "\n".join(
        [f"{i}. {cat} (упоминаний: {count})" for i, (cat, count) in enumerate(rare_categories_with_counts, start=1)]
    )

    message = (
        f"Следующие категории используются реже остальных:\n\n{rare_categories_text}\n\n"
        "Хотите ли вы удалить некоторые из этих категорий перед финальной обработкой?"
    )

    keyboard = [
        [InlineKeyboardButton("Удалить некоторые категории", callback_data="remove_rare")],
        [InlineKeyboardButton("Оставить все категории", callback_data="keep_all")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query:
        await update.callback_query.edit_message_text(text=message, reply_markup=reply_markup)
    else:
        await context.bot.send_message(chat_id=chat_id, text=message, reply_markup=reply_markup)

    logger.info("Sent rare categories list to user for removal decision")
    return RARE_CATEGORIES









async def remove_categories_input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_input = update.message.text
    rare_categories_with_counts = context.user_data.get('rare_categories_with_counts', [])

    try:
        selected_numbers = [int(num.strip()) for num in user_input.split(",")]
        selected_numbers = list(set(selected_numbers))  # Удаляем дубликаты
        
        if any(num < 1 or num > len(rare_categories_with_counts) for num in selected_numbers):
            raise ValueError("Некорректные номера категорий")
        
        categories_to_remove = [rare_categories_with_counts[i-1][0] for i in selected_numbers]
        
        df, category_to_code, code_to_category, _ = context.user_data['survey_data']
        
        # Удаляем выбранные категории
        for category in categories_to_remove:
            if category in category_to_code:
                del category_to_code[category]
        
        # Обновляем код_к_категории после удаления категорий
        code_to_category = {v: k for k, v in category_to_code.items()}
        
        await update.message.reply_text(f"Удалены категории: {', '.join(categories_to_remove)}")
        await update.message.reply_text("Начинаю пересчет результатов без удаленных категорий. Это может занять некоторое время...")

        # Пересчитываем результаты
        df, category_to_code, code_to_category = categorize_answers(
            df, context.user_data["open_answer_column"], list(category_to_code.keys())
        )
        
        df = test_categorizations(
            df, context.user_data["open_answer_column"], list(category_to_code.keys()), category_to_code, code_to_category
        )
        
        # Создаем новые столбцы категорий
        df = create_category_columns(df, list(category_to_code.keys()), category_to_code)
        
        # Повторный анализ категорий
        categories, rare_categories, common_categories = analyze_category_usage(df, list(category_to_code.keys()))
        
        # Обновляем данные в контексте
        context.user_data['survey_data'] = (df, category_to_code, code_to_category, rare_categories)
        
        await update.message.reply_text("Пересчет завершен. Перехожу к финальной обработке результатов.")
        
        # Сразу переходим к финальной обработке
        return await process_final_results(update, context)
    
    except ValueError as e:
        await update.message.reply_text(f"Ошибка: {str(e)}. Пожалуйста, введите корректные номера категорий, разделенные запятыми.")
        logger.warning(f"User provided invalid input: {user_input}")
        return REMOVE_CATEGORIES_INPUT

    except Exception as e:
        await update.message.reply_text("Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз или начните процесс заново.")
        logger.error(f"Error in remove_categories_input_handler: {str(e)}")
        return ConversationHandler.END



async def handle_rare_categories_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    if query.data == "choose_to_delete":
        return await prompt_for_categories_to_remove(update, context)
    elif query.data == "keep_all":
        return await process_final_results(update, context)

async def prompt_for_categories_to_remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = query.message.chat_id
    rare_categories_with_counts = context.user_data.get("rare_categories_with_counts", [])

    rare_categories_text = "\n".join(
        [f"{i}. {cat} (упоминаний: {count})" for i, (cat, count) in enumerate(rare_categories_with_counts, start=1)]
    )

    message = (
        f"Выберите номера категорий для удаления (через запятую):\n\n{rare_categories_text}\n\n"
        "Например: 1,3,5"
    )

    await query.edit_message_text(text=message)
    return REMOVE_CATEGORIES_INPUT




async def category_choice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    if query.data == "use_categories" or query.data == "start_analysis":
        return await process_survey_data(update, context)
    elif query.data == "increase_categories":
        # Установка max_categories в 50
        context.user_data["max_categories"] = 50
        # Повторная генерация категорий
        await query.edit_message_text("Генерирую новые категории...")
        return await suggest_categories_handler(update, context)
    elif query.data == "decrease_categories":
        # Установка max_categories в 10
        context.user_data["max_categories"] = 10
        # Повторная генерация категорий
        await query.edit_message_text("Генерирую новые категории...")
        return await suggest_categories_handler(update, context)
    elif query.data == "edit":
        # Предлагаем новые опции редактирования
        keyboard = [
            [InlineKeyboardButton("🗑️ Удалить категории", callback_data="delete_categories")],
            [InlineKeyboardButton("➕ Добавить категорию", callback_data="add_category")],
            [InlineKeyboardButton("✏️ Изменить название категории", callback_data="rename_category")],
            [InlineKeyboardButton("🚀 Запустить анализ", callback_data="finish_editing")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "Выберите действие для редактирования категорий:",
            reply_markup=reply_markup,
        )
        return EDIT_CATEGORIES
    else:
        await query.edit_message_text(
            "Неизвестный выбор. Пожалуйста, попробуйте еще раз."
        )
        return CHOOSE_CATEGORIES




async def general_callback_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    logger.debug(f"Received callback query: {query.data}")
    await query.answer()
    await update.effective_message.reply_text(f"Вы нажали кнопку: {query.data}")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    # Отправляем сообщение пользователю
    if isinstance(update, Update) and update.effective_chat:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.",
        )

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"Всего проанализировано ответов: {total_answers}",
    )
    logger.info(f"User {chat_id} requested stats. Total answers: {total_answers}.")


async def precheckout_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.pre_checkout_query
    if query.invoice_payload.startswith("paid_answers_"):
        await query.answer(ok=True)
    else:
        await query.answer(ok=False, error_message="Что-то пошло не так...")

def update_stats(user_id: str, answers_analyzed: int, money_saved: float, time_saved: float):
    """
    Обновляет глобальную и пользовательскую статистику.

    Args:
        user_id (str): ID пользователя Telegram.
        answers_analyzed (int): Количество проанализированных ответов.
        money_saved (float): Сэкономленные деньги (в рублях).
        time_saved (float): Сэкономленное время (в минутах).
    """
    try:
        # Обновление глобальной статистики
        stats_ref = ref.child('stats')
        stats_ref.child('total_answers').transaction(lambda current: (current or 0) + answers_analyzed)
        stats_ref.child('total_surveys').transaction(lambda current: (current or 0) + 1)
        stats_ref.child('total_money_saved').transaction(lambda current: (current or 0.0) + money_saved)
        stats_ref.child('total_time_saved').transaction(lambda current: (current or 0.0) + time_saved)
        
        # Обновление пользовательской статистики
        user_ref = ref.child('users').child(str(user_id))
        user_ref.child('answers_analyzed').transaction(lambda current: (current or 0) + answers_analyzed)
        user_ref.child('money_saved').transaction(lambda current: (current or 0.0) + money_saved)
        user_ref.child('time_saved').transaction(lambda current: (current or 0.0) + time_saved)
        
        logger.info(f"Обновлена статистика для пользователя {user_id}: +{answers_analyzed} ответов, +{money_saved} руб., +{time_saved} мин.")
    except Exception as e:
        logger.error(f"Не удалось обновить статистику для пользователя {user_id}: {e}")
        raise e  # Повторно выбрасываем исключение для обработки в вызывающем коде





async def process_final_results(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    df, category_to_code, code_to_category, _ = context.user_data['survey_data']
    total_answers = len(df)

    user_data = get_user_data(user_id)
    used_free_answers = user_data.get('free_answers', 0)
    remaining_free_answers = max(0, 2000 - used_free_answers)  # Обновлено до 2000
    free_answers = min(remaining_free_answers, total_answers)
    paid_answers = total_answers - free_answers

    if free_answers > 0:
        new_free_answers = used_free_answers + free_answers
        update_user_free_answers(user_id, new_free_answers)

        free_df = df.head(free_answers)
        xlsx_content, png_content = save_results(free_df, code_to_category)
        await send_results(update, context, xlsx_content, png_content)

        if paid_answers > 0:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Обработано {free_answers} бесплатных ответов. Осталось {paid_answers} платных ответов."
            )
            await send_payment_request(update, context, paid_answers)
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Анализ завершен. Все ответы обработаны бесплатно."
            )
    elif paid_answers > 0:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"У вас закончились бесплатные ответы. Необходимо оплатить анализ {paid_answers} ответов."
        )
        await send_payment_request(update, context, paid_answers)

    context.user_data['paid_answers'] = paid_answers

    # Расчет экономии
    agency_cost_per_answer = 5  # рублей
    bot_cost_per_answer = 2     # рубля
    agency_time_per_answer_min = 0.5  # 30 секунд
    bot_time_per_1000_answers_min = 5   # 5 минут
    bot_time_per_answer_min = bot_time_per_1000_answers_min / 1000  # 0.005 минут

    money_saved = (agency_cost_per_answer - bot_cost_per_answer) * total_answers
    time_saved_min = (agency_time_per_answer_min - bot_time_per_answer_min) * total_answers

    # Обновление статистики
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, update_stats, user_id, total_answers, money_saved, time_saved_min)
    except Exception as e:
        logger.error(f"Не удалось обновить статистику для пользователя {user_id}: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Произошла ошибка при обновлении статистики. Пожалуйста, попробуйте позже."
        )
        return ConversationHandler.END

    await context.bot.send_message(
        chat_id=chat_id,
        text="Спасибо за использование бота!"
    )
    return ConversationHandler.END






async def stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    try:
        # Получение глобальной статистики
        stats_ref = ref.child('stats')
        total_answers = stats_ref.child('total_answers').get()
        total_surveys = stats_ref.child('total_surveys').get()
        total_money_saved = stats_ref.child('total_money_saved').get()
        total_time_saved_min = stats_ref.child('total_time_saved').get()

        if total_answers is None:
            total_answers = 0
        if total_surveys is None:
            total_surveys = 0
        if total_money_saved is None:
            total_money_saved = 0.0
        if total_time_saved_min is None:
            total_time_saved_min = 0.0

        # Получение статистики пользователя
        user_stats = get_user_stats(user_id)
        user_answers = user_stats.get('answers_analyzed', 0)
        user_money_saved = user_stats.get('money_saved', 0.0)
        user_time_saved_min = user_stats.get('time_saved', 0.0)

        # Форматирование чисел
        total_answers_str = f"{total_answers:,}".replace(",", " ")
        total_surveys_str = f"{total_surveys:,}".replace(",", " ")
        total_money_saved_str = f"{total_money_saved:,.2f} руб."
        if total_time_saved_min >= 60:
            hours = total_time_saved_min // 60
            minutes = total_time_saved_min % 60
            total_time_saved_str = f"{int(hours)} часов {int(minutes)} минут"
        else:
            total_time_saved_str = f"{total_time_saved_min:.2f} минут"

        user_money_saved_str = f"{user_money_saved:,.2f} руб."
        if user_time_saved_min >= 60:
            hours = user_time_saved_min // 60
            minutes = user_time_saved_min % 60
            user_time_saved_str = f"{int(hours)} часов {int(minutes)} минут"
        else:
            user_time_saved_str = f"{user_time_saved_min:.2f} минут"

        # Формирование сообщения статистики
        stats_message = (
            f"📊 *Статистика:*\n\n"
            f"🔹 *Общая статистика:*\n"
            f"   - Всего проанализировано ответов: *{total_answers_str}*\n"
            f"   - Общее количество использований бота: *{total_surveys_str}*\n"
            f"   - Всего сэкономлено денег: *{total_money_saved_str}*\n"
            f"   - Всего сэкономлено времени: *{total_time_saved_str}*\n\n"
            f"🔸 *Ваша статистика:*\n"
            f"   - Вы проанализировали ответов: *{user_answers}*\n"
            f"   - Вы сэкономили денег: *{user_money_saved_str}*\n"
            f"   - Вы сэкономили времени: *{user_time_saved_str}*\n\n"
            f"💡 *Как это посчитано:*\n"
            f"   - *Экономия денег:* (5 руб. - 2 руб.) × количество ответов.\n"
            f"   - *Экономия времени:* (0.5 мин. - 0.005 мин.) × количество ответов.\n"
        )

        await context.bot.send_message(
            chat_id=chat_id,
            text=stats_message,
            parse_mode='Markdown'
        )
        logger.info(f"Пользователь {chat_id} запросил статистику. Всего ответов: {total_answers}, Всего использований: {total_surveys}.")
    except Exception as e:
        logger.error(f"Не удалось получить статистику из Firebase: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Произошла ошибка при получении статистики. Пожалуйста, попробуйте позже."
        )






def get_total_answers() -> int:
    """
    Получает общее количество проанализированных ответов из Firebase.
    """
    stats_ref = ref.child('stats').child('total_answers')
    total = stats_ref.get()
    if total is None:
        return 0
    return total

def get_total_surveys() -> int:
    """
    Получает общее количество использований бота из Firebase.
    """
    surveys_ref = ref.child('stats').child('total_surveys')
    total = surveys_ref.get()
    if total is None:
        return 0
    return total

def get_user_stats(user_id: str) -> dict:
    """
    Получает статистику для конкретного пользователя.
    """
    user_ref = ref.child('users').child(str(user_id))
    user_stats = user_ref.get()
    if user_stats is None:
        return {
            'answers_analyzed': 0,
            'money_saved': 0.0,
            'time_saved': 0.0
        }
    return user_stats




def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # **Добавляем обработчики команд /payment и /help**
    application.add_handler(CommandHandler("payment", payment_handler))
    application.add_handler(CommandHandler("help", help_handler))

    # Создаем ConversationHandler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            UPLOAD: [
                CommandHandler("start", start),
                MessageHandler(filters.Document.ALL, file_handler)
            ],
            QUESTION: [
                CommandHandler("start", start),
                MessageHandler(filters.TEXT & ~filters.COMMAND, question_handler)
            ],
            COLUMN: [
                CommandHandler("start", start),
                MessageHandler(filters.TEXT & ~filters.COMMAND, column_handler)
            ],
            CATEGORIES: [
                CommandHandler("start", start),
                CallbackQueryHandler(categories_handler),
                MessageHandler(filters.TEXT & ~filters.COMMAND, manual_categories_handler),
            ],
            CHOOSE_CATEGORIES: [
                CommandHandler("start", start),
                CallbackQueryHandler(category_choice_handler)
            ],
            EDIT_CATEGORIES: [
                CommandHandler("start", start),
                CallbackQueryHandler(edit_categories_handler),
            ],
            DELETE_CATEGORIES: [
                CommandHandler("start", start),
                MessageHandler(filters.TEXT & ~filters.COMMAND, delete_categories_handler)
            ],
            ADD_CATEGORY: [
                CommandHandler("start", start),
                MessageHandler(filters.TEXT & ~filters.COMMAND, add_category_handler)
            ],
            RENAME_CATEGORY: [
                CommandHandler("start", start),
                MessageHandler(filters.TEXT & ~filters.COMMAND, rename_category_handler)
            ],
            PROCESSING: [
                CommandHandler("start", start),
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_survey_data)
            ],
            RARE_CATEGORIES: [
                CallbackQueryHandler(handle_rare_categories_callback),
                CommandHandler("cancel", cancel)
            ],
            REMOVE_CATEGORIES_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, remove_categories_input_handler),
                CommandHandler("cancel", cancel)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    # Добавляем ConversationHandler
    application.add_handler(conv_handler)
    
    # **Добавляем обработчик команды /stats**
    application.add_handler(CommandHandler("stats", stats_handler))
    
    # Добавляем существующие обработчики
    application.add_handler(CommandHandler("free_answers", check_free_answers))
    application.add_handler(PreCheckoutQueryHandler(precheckout_callback))
    application.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_callback))

    # Добавляем обработчик ошибок
    application.add_error_handler(error_handler)

    # Запуск бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)








if __name__ == "__main__":
    main()
