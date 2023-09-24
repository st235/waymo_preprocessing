import os


def list_all_files_with_extension(files: list,
                                  extension: str) -> list:
    result = list()

    for file in files:
        if os.path.isfile(file) and file.endswith(extension):
            result.append(file)
        elif os.path.isdir(file):
            sub_files = [os.path.join(file, f) for f in os.listdir(file)]
            result.extend(list_all_files_with_extension(sub_files, extension))

    return result
