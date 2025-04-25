import os
import re
import datetime

def make_dirs(path):
    os.makedirs(path, exist_ok=True)

def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else -1

def init_log(log_file_path):
    with open(log_file_path, 'w') as f:
        f.write(f"==== Batch Image Processing Log ====\nStart time: {datetime.datetime.now()}\n\n")

def write_log(log_file_path, message):
    with open(log_file_path, 'a') as f:
        f.write(message + "\n")
    print(message)
