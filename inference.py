import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import seg_metrics.seg_metrics as sg
from _main_.main import *
import csv
import nibabel as nib

def crop_center(image, crop_size):
    h, w = image.shape
    start_x = w // 2 - (crop_size // 2)
    start_y = h // 2 - (crop_size // 2)
    return image[start_y:start_y+crop_size, start_x:start_x+crop_size]

# For IRCAD, either for Patient9.json or Patient6.json
def datafold_test_IRCAD(datalist, basedir, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    test = []

    for d in json_data[key]:
        image_path = os.path.join(basedir, d["image"])
        label_path = os.path.join(basedir, d["label"])
        test.append({"image": image_path, "label": label_path})

    return test

# For PICAI test.json, were fold_num is the testing patient number
def datafold_test_PICAI(datalist, basedir, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    test= []
    for d in json_data:
        if "fold" in d:
            fold_num = d["fold"]
            if fold_num == 10322: # 20 test cases
                test.append(d)

    return test
