import base64
import json
import mimetypes
import requests
import PIL
from openai import OpenAI
from pydantic import BaseModel, Field
import io
import re

from groq import Groq


class EvaluationResult(BaseModel):
    score: str = Field(description="Оценка работы")
    comment: str = Field(description="Общий развернутый комментарий")
    errors: str = Field(description="Список найденных ключевых ошибок")


class AIProvider:
    def __init__(self, provider_name: str, api_key: str):
        if not api_key:
            raise ValueError("Ошибка: API-ключ не может быть пустым.")

        self.client = Groq(api_key=api_key)
        # Указываем конкретную БЕСПЛАТНУЮ Gemini, которая умеет читать и PDF, и JPEG
        self.model_name = provider_name

    def get_models(self) -> list[str]:
        models = [model.id for model in self.client.models.list().data]
        return models

    def get_vision_models(self) -> list[str]:
        models = self.get_models()
        def check_vision_support(model_id):
            """Проверяет, умеет ли модель работать с изображениями"""
            TINY_IMAGE_BASE64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

            try:
                self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "тест"},
                                {"type": "image_url", "image_url": {"url": TINY_IMAGE_BASE64}}
                            ]
                        }
                    ],
                    max_tokens=1  # Запрашиваем всего 1 токен, чтобы не тратить лимиты
                )
                return True
            except Exception as e:
                # Если API ругается, что модель не поддерживает изображения, возвращаем False
                if "image" in str(e).lower() or "multimodal" in str(e).lower():
                    return False
                # На случай других ошибок (например, временный сбой модели), тоже пропускаем
                return False

        vision_models = []
        for model in models:
            if check_vision_support(model):
                vision_models.append(model)
        return vision_models

    def _detect_text(self, img, model_name="meta-llama/llama-4-scout-17b-16e-instruct") -> str:
        prompt_text = (
            "Ты — специализированная система распознавания текста (OCR) высочайшего уровня. "
            "Твоя единственная задача — переписать весь рукописный и печатный текст с предоставленного изображения. "
            "СТРОГИЕ ПРАВИЛА И ОГРАНИЧЕНИЯ:\n"
            "1. Выводи только чистый распознанный текст. Никаких вводных фраз типа 'Вот текст:', комментариев, пояснений или примечаний.\n"
            "2. Пиши строго на русском языке (если в тексте встречаются латинские обозначения заданий, например B2, B3 — оставь их латиницей).\n"
            "3. Полностью игнорируй тетрадные клетки, линейки, поля и любой фоновый шум. Не пытайся описывать их.\n"
            "4. Сохраняй исходное разбиение на абзацы и структуру предложений. Если текст написан в колонки — читай их последовательно (сначала левую, затем правую).\n"
            "5. Если какое-то слово написано неразборчиво, восстанови его по смыслу контекста. Не ставь знаки вопроса или прочерки вместо неразборчивых букв.\n"
            "6. Не исправляй орфографические и грамматические ошибки автора — переписывай текст точно так, как он написан в оригинале.\n"
            "7. Сохраняй авторские знаки препинания и сокращения (например: т.к., т.п.)."
        )

        # Кодируем картинку
        base64_image = base64.b64encode(
            (b := io.BytesIO(), img.convert("RGB").save(b, "JPEG"), b.getvalue())[2]).decode()

        try:
            chat_completion = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_text
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.0,  # Минимальная температура для более точного распознавания
            )

            # Возвращаем ответ модели
            return chat_completion.choices[0].message.content

        except Exception as e:
            return f"Ошибка при анализе изображения: {e}"

    def evaluate_text(self, text: str, prompt: str) -> dict:
        system_prompt = prompt + '\n' + (
            "Ты — эксперт по проверке учебных работ на русском языке. Не торопись, мне нужен качественный ответ."
            "Проанализируй текст и верни ответ СТРОГО в формате JSON. "
            "Используй только следующие ключи (названия должны совпадать один в один):\n"
            "{\n"
            '  "Оценка": "",\n'
            '  "Комментарий": "полный разбор работы, сильные и слабые стороны, распиши все очень подробно",\n'
            '  "Ошибки": "перечень ошибок по смыслу или несоответствие критериям, если они есть, '
            'грамматические и пунктуационные не учитывай, тоже распиши подробно, в json с ответом передай их '
            'не в виде списка, а просто текстом",\n'
            '  "Вероятность выполнения ИИ": "Высокая, Средняя или Низкая, обоснуй кратко присовенную вероятность"\n'
            "}"
        )

        try:
            # Отправляем запрос в Groq
            completion = self.client.chat.completions.create(
                # Используем одну из лучших доступных моделей (Llama 3.1 70B или актуальную на данный момент)
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Текст для проверки:\n{text}"}
                ],
                # ВКЛЮЧАЕМ JSON MODE (Обязательное требование Groq при этом — слово JSON в системном промпте)
                response_format={"type": "json_object"},
                temperature=0.2
            )

            # Получаем чистую JSON строку
            raw_content = completion.choices[0].message.content
            return json.loads(raw_content)

        except Exception as e:
            print(f"Ошибка при проверке работы: {e}")
            return None
