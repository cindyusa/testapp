import os

def read_txt_files(root_dir):
    text_data = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_data.append({
                        "subdir": subdir,
                        "filename": file,
                        "content": f.read()
                    })
    return text_data