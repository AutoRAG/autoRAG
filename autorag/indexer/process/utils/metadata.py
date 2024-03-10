import json


def metadata_get_url_from_header(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        return json.loads(f.readline())


file_metadata_dict = {"get_url_from_header": metadata_get_url_from_header}
