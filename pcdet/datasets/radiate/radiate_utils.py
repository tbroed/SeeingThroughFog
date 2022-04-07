import json

from pathlib import Path
from pprint import pprint
from typing import Dict, List


def get_annoations(path: Path) -> List:

    with open(path) as f:
        return json.load(f)


def get_num_valid_annoations(annos: List) -> Dict:

    count_dict = {}

    for anno in annos:

        bboxes = anno['bboxes']

        for bbox in bboxes:

            if type(bbox) is dict:

                if bbox['position']:

                        try:
                            count_dict[anno['class_name']] += 1
                        except KeyError:
                            count_dict[anno['class_name']] = 1

    return count_dict


if __name__ == '__main__':

    root = Path.home() / 'datasets/Radiate'

    splits = [('snowy', 'snow_1_0'), ('foggy', 'tiny_foggy/tiny_foggy')]

    for name, subpath in splits:

        hard_coded = Path.home() / 'datasets/Radiate' / subpath / 'annotations' / 'annotations.json'

        annotations = get_annoations(path=hard_coded)

        print()
        print(f'{len(annotations)} {name} frames')
        pprint(get_num_valid_annoations(annos=annotations))