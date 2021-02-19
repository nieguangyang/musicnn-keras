import os

# path
PATH = "/".join(os.path.realpath(__file__).replace("\\", "/").split("/")[:-1])
MTT_TAG_CSV = PATH + "/mtt_tags.csv"
MSD_TAG_CSV = PATH + "/msd_tags.csv"


def tags_from_csv(csv_file):
    """
    :param csv_file: str, MTT_TAG_CSV or MSD_TAG_CSV
    :return tags: list, [tag_info, ...]
        tag_info: dict, tag information, e.g.
            {"tag": "classical", "category": "genre", "indices": [1, 34], "original_tags": ["classical", "classic"]}
            tag: str, unified tag
            category: str, category of tag
            indices: list, indices of corresponding model's output
            original_tags: list, corresponding original tags
    """
    if csv_file == MTT_TAG_CSV:
        source = "mtt"
    elif csv_file == MSD_TAG_CSV:
        source = "msd"
    else:
        raise Exception("wrong csv_file given")
    tags = {}
    f = open(csv_file, "r")
    f.readline()
    for i, line in enumerate(f):
        original, unified, category = line.rstrip("\n").split(",")[:3]
        tag_info = tags.setdefault(unified, dict(tag=unified, source=source, category=category))
        indices = tag_info.setdefault("indices", [])
        indices.append(i)
        original_tags = tag_info.setdefault("original_tags", [])
        original_tags.append(original)
    f.close()
    tags = list(tags.values())
    tags.sort(key=lambda t: t["category"])  # sort by category
    return tags


MTT_TAGS = tags_from_csv(MTT_TAG_CSV)
MSD_TAGS = tags_from_csv(MSD_TAG_CSV)


def display_tags():
    print("MTT tags")
    for tag_info in MTT_TAGS:
        print(tag_info)
    print("MSD tags")
    for tag_info in MSD_TAGS:
        print(tag_info)


if __name__ == "__main__":
    display_tags()
