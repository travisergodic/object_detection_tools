import os
import random
import shutil
import argparse
from pathlib import Path
import sys
sys.path.insert(0, os.getcwd())

from src.utils import get_image_anno_pairs

RANDOM_SEED = 14
TRAIN_FOLDER = 'train'
TEST_FOLDER = 'test'
ANNO_SUFFIX = '.txt'


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

    print(f"Training set has {len(train_image_anno_pair_list)} image-annotation pairs")
    print(f"Testing set has {len(test_image_anno_pair_list)} image-annotation pairs")

    # mkdirs
    (Path(args.root) / TRAIN_FOLDER).mkdir(parents=True, exist_ok=False)
    (Path(args.root) / TEST_FOLDER).mkdir(parents=True, exist_ok=False)
    (Path(args.root) / TRAIN_FOLDER / 'images').mkdir(parents=True, exist_ok=True)
    (Path(args.root) / TRAIN_FOLDER / 'labels').mkdir(parents=True, exist_ok=True)
    (Path(args.root) / TEST_FOLDER / 'images').mkdir(parents=True, exist_ok=True)
    (Path(args.root) / TEST_FOLDER / 'labels').mkdir(parents=True, exist_ok=True)

    # copy image & annoation from src to dst
    for image_path, anno_path in train_image_anno_pair_list:
        shutil.copy(str(image_path), str(Path(args.root) / TRAIN_FOLDER / 'images' / image_path.name))
        shutil.copy(str(anno_path), str(Path(args.root) / TRAIN_FOLDER / 'labels' / anno_path.name))
        
    for image_path, anno_path in test_image_anno_pair_list:
        shutil.copy(str(image_path), str(Path(args.root) / TEST_FOLDER / 'images' / image_path.name))
        shutil.copy(str(anno_path), str(Path(args.root) / TEST_FOLDER / 'labels' / anno_path.name))


parser = argparse.ArgumentParser(description="Train test split for YOLO format data.")
parser.add_argument("--img_folder", type=str)
parser.add_argument("--anno_folder", type=str)
parser.add_argument("--img_suffix_list", nargs='+')
parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--root")
args = parser.parse_args()
main() 