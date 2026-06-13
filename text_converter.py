import os
import io
from PIL import Image
import easyocr
import pypdf
from PIL import ImageEnhance, ImageOps
import numpy as np

from ai_client import AIProvider


class SimpleTextExtractor:
    def __init__(self, api_key: str, model_name=None):
        self.ai = AIProvider("pass", api_key)
        if (model_name is None) or ('vision' not in model_name.lower()):
            model_name = None
        self.model_name = model_name

    def extract_text(self, file_path) -> str:
        """Основной метод обработки."""
        if not os.path.exists(file_path):
            return f"Ошибка: Файл {file_path} не найден."

        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return self._process_pdf(file_path)
        elif ext in ['.jpg', '.jpeg', '.png']:
            return self._process_image_path(file_path)
        else:
            return f"Ошибка: Неподдерживаемый формат ({ext})."

    def _process_image_path(self, img_path) -> str:
        """Распознавание текста с предварительным улучшением картинки."""
        try:
            # Открываем изображение через Pillow
            img = Image.open(img_path)
        except Exception as e:
            return f"Ошибка открытия картинки: {e}"
        try:
            if self.model_name is None:
                text = self.ai._detect_text(img)
            else:
                text = self.ai._detect_text(img, self.model_name)
        except Exception as e:
            return f"Ошибка чтения картинки в ИИ: {str(e)}"
        return text

    def _process_pdf(self, pdf_path):
        """Обработка PDF: чтение печатного текста + OCR картинок внутри без Poppler."""
        try:
            reader = pypdf.PdfReader(pdf_path)
            pdf_text = []

            for i, page in enumerate(reader.pages):
                page_output = []

                # 1. Пробуем извлечь встроенный печатный текст (быстро и точно)
                text = page.extract_text()
                if text and text.strip():
                    page_output.append(text.strip())

                # 2. Извлекаем картинки (сканы/фото), если они есть на странице
                for image_file_object in page.images:
                    try:
                        # Читаем байты картинки прямо в память
                        image_bytes = image_file_object.data
                        img = Image.open(io.BytesIO(image_bytes))
                        if self.model_name is None:
                            text = self.ai._detect_text(img)
                        else:
                            text = self.ai._detect_text(img, self.model_name)
                        page_output.append(text)
                    except Exception as img_err:
                        return img_err

                # Собираем текст страницы воедино
                combined_page_text = "\n".join(page_output)
                pdf_text.append(f"--- Страница {i + 1} ---\n{combined_page_text}")

            return "\n\n".join(pdf_text)
        except Exception as e:
            return f"Ошибка обработки PDF: {str(e)}"

