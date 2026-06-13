import os


def is_work_available(file_path: str) -> bool:
    return os.path.isfile(file_path) and os.path.splitext(file_path)[1] in ['.pdf', '.jpg', '.jpeg', '.png']


def extract_works(folder_path: str) -> list[str]:
    ans = []
    for obj in os.listdir(folder_path):
        file_path = os.path.join(folder_path, obj)
        if is_work_available(file_path):
            ans.append(file_path)
    return ans


