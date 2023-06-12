import os
import random
import shutil
import argparse
from pathlib import Path
import sys
sys.path.insert(0, os.getcwd())

from src.utils import get_image_anno_pairs

RANDOM_SEED = 14
IMAGE_FOLDER = 'VOC2007/JPEGImages'
ANNOTATION_FOLDER = 'VOC2007/Annotations'
TRAIN_TEST_FOLDER = 'VOC2007/ImageSets/Main'
ANNO_SUFFIX = '.xml'


def main():
    random.seed(RANDOM_SEED)
    # collect
    image_anno_pair_list = get_image_anno_pairs(args.img_folder, args.anno_folder, args.img_suffix_list, ANNO_SUFFIX)
    length = len(image_anno_pair_list)  
    print(f"Found {length} image-annotation pairs.")

    # train test split
    random.shuffle(image_anno_pair_list)
    train_image_anno_pair_list = image_anno_pair_list[
        :int(length * args.train_ratio)
    ]
    test_image_anno_pair_list = image_anno_pair_list[
        int(length * args.train_ratio):(int(length * args.train_ratio) + int(length * (1 - args.train_ratio)))
    ]

    # mkdirs
    (Path(args.root) / IMAGE_FOLDER).mkdir(parents=True, exist_ok=False)
    (Path(args.root) / ANNOTATION_FOLDER).mkdir(parents=True, exist_ok=False)
    (Path(args.root) / TRAIN_TEST_FOLDER).mkdir(parents=True, exist_ok=False)

    # create train test files
    with open(str(Path(args.root) / TRAIN_TEST_FOLDER / 'trainval.txt'), 'w') as f:
        f.write(
            '\n'.join([image_anno_pair[0].name for image_anno_pair in train_image_anno_pair_list])
        )

    with open(str(Path(args.root) / TRAIN_TEST_FOLDER / 'test.txt'), 'w') as f:
        f.write(
            '\n'.join([image_anno_pair[0].name for image_anno_pair in test_image_anno_pair_list])
        )

    print(f"Training set has {len(train_image_anno_pair_list)} image-annotation pairs")
    print(f"Testing set has {len(test_image_anno_pair_list)} image-annotation pairs")

    # copy image & annoation from src to dst
    for image_path, anno_path in image_anno_pair_list:
        shutil.copy(str(image_path), str(Path(args.root) / IMAGE_FOLDER / image_path.name))
        shutil.copy(str(anno_path), str(Path(args.root) / ANNOTATION_FOLDER / anno_path.name))


parser = argparse.ArgumentParser(description="Train test split for VOC format dataset.")
parser.add_argument("--img_folder", type=str)
parser.add_argument("--anno_folder", type=str)
parser.add_argument("--img_suffix_list", nargs='+')
parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--root")
args = parser.parse_args()
main() 