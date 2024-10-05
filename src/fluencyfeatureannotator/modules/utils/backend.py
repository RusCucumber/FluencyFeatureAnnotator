from typing import Union

from werkzeug.datastructures import FileStorage


def check_file(file: FileStorage, extentions: list) -> Union[bool, str]:
    if not isinstance(file, FileStorage):
        raise TypeError(f"argument file must be werkzeug.datastructures.FileStorage, not {type(file)}")
    if not isinstance(extentions, list):
        raise TypeError(f"argument extentions must be list, not{type(list)}")

    # check if file is uploaded
    if file.filename == "":
        return False, "file was not uploaded"

    # check filename extentions
    extention = file.filename.rsplit(".", 1)[1]
    if extention in extentions:
        return True, "OK"

    return False, f"file extention must be \"{', '.join(extentions)}\", \"{extention}\" not allowed"
