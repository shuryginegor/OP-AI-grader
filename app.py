import os
import sys
import io
import tkinter as tk
from tkinter import filedialog
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd

# Импорт PyQt6
from PyQt6.QtCore import QUrl, QThread
from PyQt6.QtGui import QIcon
from PyQt6.QtWebEngineCore import QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication

# Импорт ваших существующих модулей
from ai_client import AIProvider
from text_converter import SimpleTextExtractor
from processing_of_works import is_work_available, extract_works

# Определение базовой папки шаблонов
BASE_DIR = sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_FOLDER)

progress_status = {
    "status": "idle",
    "current": 0,
    "total": 0,
    "percentage": 0,
    "data_frame": None,
    "error_message": ""
}


# --- ВАШИ СТАНДАРТНЫЕ ФУНКЦИИ И РОУТЫ FLASK ---

def process_tasks_thread(api_key, text_model, prompt_user, folder_path):
    global progress_status
    try:
        all_files = extract_works(folder_path)
        valid_files = [f for f in all_files if is_work_available(f)]
        total_files = len(valid_files)

        if total_files == 0:
            progress_status.update({"status": "error", "error_message": "Файлы не найдены."})
            return

        progress_status.update({"status": "processing", "current": 0, "total": total_files, "percentage": 0})

        extractor = SimpleTextExtractor(api_key=api_key)
        ai_client = AIProvider(provider_name=text_model, api_key=api_key)

        full_prompt = f"{prompt_user}\n\nОтвет предоставь строго в формате JSON..."
        results = []

        for idx, file_path in enumerate(valid_files):
            file_name = os.path.basename(file_path)
            try:
                extracted_text = extractor.extract_text(file_path)
                ai_response = ai_client.evaluate_text(text=extracted_text, prompt=full_prompt)
                row_data = {"Файл": file_name}
                row_data.update(ai_response)
                results.append(row_data)
            except Exception as e:
                results.append({"Файл": file_name, "Оценка": "Ошибка", "Комментарий": str(e)})

            current_count = idx + 1
            progress_status["current"] = current_count
            progress_status["percentage"] = int((current_count / total_files) * 100)

        df = pd.DataFrame(results)
        progress_status.update({"status": "completed", "data_frame": df})
    except Exception as e:
        progress_status.update({"status": "error", "error_message": str(e)})


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/verify-key', methods=['POST'])
def verify_key():
    data = request.get_json() or {}
    api_key = data.get('api_key') or request.form.get('api_key')

    if not api_key:
        return jsonify({"success": False, "message": "Ключ не может быть пустым."}), 400

    try:
        provider = AIProvider(provider_name="grok-beta", api_key=api_key)
        text_models = provider.get_models()
        return jsonify({"success": True, "text_models": text_models})
    except Exception as e:
        return jsonify({"success": False, "message": f"Ошибка авторизации: {str(e)}"})


@app.route('/select-folder', methods=['POST'])
def select_folder():
    try:
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)
        folder_selected = filedialog.askdirectory(parent=root)
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

    import threading
    threading.Thread(target=process_tasks_thread, args=(
        request.form.get('api_key'), request.form.get('text_model'),
        request.form.get('prompt'), request.form.get('folder_path')
    )).start()
    return jsonify({"success": True})


@app.route('/progress-page')
def progress_page():
    return render_template('progress.html')


@app.route('/status')
def get_status():
    status_copy = {k: v for k, v in progress_status.items() if k != "data_frame"}
    return jsonify(status_copy)


@app.route('/download')
def download_file():
    global progress_status
    if progress_status["status"] == "completed" and progress_status["data_frame"] is not None:
        excel_buffer = io.BytesIO()
        progress_status["data_frame"].to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        return send_file(
            excel_buffer,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name="результаты_проверки.xlsx"
        )
    return "Файл не найден", 404


@app.route('/reset')
def reset_status():
    progress_status.update(
        {"status": "idle", "current": 0, "total": 0, "percentage": 0, "data_frame": None, "error_message": ""})
    return jsonify({"success": True})


# --- ИЗОЛИРОВАННЫЙ ПОТОК QT ДЛЯ FLASK ---

class FlaskServerThread(QThread):
    def run(self):
        # Запуск сервера Flask без использования сторонних процессов
        app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


# --- ГРАФИЧЕСКОЕ ОКНО ПРИЛОЖЕНИЯ ---

class MainWindow(QWebEngineView):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Проверка работ ИИ")
        self.resize(1024, 768)

        icon_path = os.path.join(TEMPLATE_FOLDER, "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        QWebEngineProfile.defaultProfile().downloadRequested.connect(self.handle_download)
        self.load(QUrl("http://127.0.0.1:5000"))

    def handle_download(self, download_item):
        downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        download_item.setDownloadDirectory(downloads_dir)
        download_item.setDownloadFileName(download_item.suggestedFileName())
        download_item.accept()

    def closeEvent(self, event):
        # Принудительное закрытие всех фоновых процессов и ОЗУ при нажатии на крестик
        os._exit(0)


if __name__ == '__main__':
    qt_app = QApplication(sys.argv)

    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("my_ai_checker_app")

    # Безопасный и чистый запуск потока сервера внутри Qt-архитектуры
    flask_thread = FlaskServerThread()
    flask_thread.start()

    window = MainWindow()
    window.show()
    sys.exit(qt_app.exec())
