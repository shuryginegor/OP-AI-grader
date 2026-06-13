import os
import threading
import tkinter as tk
from tkinter import filedialog
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import sys
import threading
import webbrowser
# Импорт ваших существующих модулей
from ai_client import AIProvider
from text_converter import SimpleTextExtractor
from processing_of_works import is_work_available, extract_works

if hasattr(sys, "_MEIPASS"):
    base_dir = sys._MEIPASS
else:
    base_dir = os.path.abspath(".")

app = Flask(__name__, template_folder=os.path.join(base_dir, "templates"))

# Глобальное состояние для отслеживания прогресса
progress_status = {
    "status": "idle",  # idle, processing, completed, error
    "current": 0,
    "total": 0,
    "percentage": 0,
    "excel_path": "",
    "error_message": ""
}


def process_tasks_thread(api_key, text_model, prompt_user, folder_path):
    global progress_status
    try:
        # 1. Поиск доступных файлов в выбранной папке
        all_files = extract_works(folder_path)
        valid_files = [f for f in all_files if is_work_available(f)]

        total_files = len(valid_files)
        if total_files == 0:
            progress_status.update({
                "status": "error",
                "error_message": "В указанной папке не найдено подходящих файлов (PDF/JPG)."
            })
            return

        progress_status.update({
            "status": "processing",
            "current": 0,
            "total": total_files,
            "percentage": 0
        })

        # 2. Инициализация ваших классов (Vision-модель не передаем, работает по умолчанию)
        extractor = SimpleTextExtractor(api_key=api_key)
        ai_client = AIProvider(provider_name=text_model, api_key=api_key)

        technical_prompt = (
            "\n\nОтвет предоставь строго в формате JSON со следующими ключами: "
            "'Оценка', 'Комментарий', 'Ошибки', 'Вероятность написания ИИ'."
        )
        full_prompt = f"{prompt_user}{technical_prompt}"

        results = []

        # 3. Обработка документов по одному
        for idx, file_path in enumerate(valid_files):
            file_name = os.path.basename(file_path)
            try:
                extracted_text = extractor.extract_text(file_path)
                ai_response = ai_client.evaluate_text(text=extracted_text, prompt=full_prompt)

                row_data = {"Файл": file_name}
                row_data.update(ai_response)
                results.append(row_data)

            except Exception as e:
                results.append({
                    "Файл": file_name,
                    "Оценка": "Ошибка",
                    "Комментарий": f"Не удалось обработать файл: {str(e)}",
                    "Ошибки": "Да",
                    "Вероятность написания ИИ": "Неизвестно"
                })

            current_count = idx + 1
            progress_status["current"] = current_count
            progress_status["percentage"] = int((current_count / total_files) * 100)

        # 4. Формирование Excel-файла
        output_filename = "результаты_проверки.xlsx"
        df = pd.DataFrame(results)
        df.to_excel(output_filename, index=False)

        progress_status.update({
            "status": "completed",
            "excel_path": output_filename
        })

    except Exception as e:
        progress_status.update({
            "status": "error",
            "error_message": f"Критическая ошибка: {str(e)}"
        })


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/verify-key', methods=['POST'])
def verify_key():
    data = request.get_json()
    api_key = data.get('api_key')

    if not api_key:
        return jsonify({"success": False, "message": "Ключ не может быть пустым."}), 400

    try:
        # Инициализируем провайдер с введенным ключом для проверки работоспособности
        provider = AIProvider(provider_name="grok-beta", api_key=api_key)

        # Запрашиваем только текстовые модели
        text_models = provider.get_models()

        return jsonify({
            "success": True,
            "text_models": text_models
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Ошибка авторизации Groq или получения моделей: {str(e)}"
        })


@app.route('/select-folder', methods=['POST'])
def select_folder():
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        folder_selected = filedialog.askdirectory()
        root.destroy()

        if folder_selected:
            return jsonify({"success": True, "folder_path": folder_selected})
        return jsonify({"success": False, "message": "Папка не выбрана"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/start', methods=['POST'])
def start_processing():
    global progress_status
    if progress_status["status"] == "processing":
        return jsonify({"success": False, "message": "Процесс уже запущен."})

    api_key = request.form.get('api_key')
    text_model = request.form.get('text_model')
    prompt_user = request.form.get('prompt')
    folder_path = request.form.get('folder_path')

    threading.Thread(target=process_tasks_thread, args=(
        api_key, text_model, prompt_user, folder_path
    )).start()

    return jsonify({"success": True})


@app.route('/progress-page')
def progress_page():
    return render_template('progress.html')


@app.route('/status')
def get_status():
    global progress_status
    return jsonify(progress_status)


@app.route('/download')
def download_file():
    global progress_status
    if progress_status["status"] == "completed" and os.path.exists(progress_status["excel_path"]):
        return send_file(progress_status["excel_path"], as_attachment=True)
    return "Файл не найден", 404


@app.route('/reset')
def reset_status():
    global progress_status
    progress_status.update({
        "status": "idle",
        "current": 0,
        "total": 0,
        "percentage": 0,
        "excel_path": "",
        "error_message": ""
    })
    return jsonify({"success": True})


def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == '__main__':
    # Запуск таймера, который откроет браузер через 1.5 секунды в фоне
    threading.Timer(1.5, open_browser).start()

    # Запуск Flask сервера (debug обязательно должен быть False!)
    app.run(host="127.0.0.1", port=5000, debug=False)
