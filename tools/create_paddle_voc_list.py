import os
import argparse
from pathlib import Path
import sys
sys.path.insert(0, os.getcwd())
from src.utils import get_image_anno_pairs


IMAGE_FOLDER = 'VOCdevkit/VOC2007/JPEGImages'
ANNOTATION_FOLDER = 'VOCdevkit/VOC2007/Annotations'


def create_paddle_voc_text(src_txt_path):
    with open(src_txt_path, 'r') as f:
        image_names = [line.strip() for line in f.readlines()]

    text = ''
    for image_name in image_names:
        img_path = str(Path(IMAGE_FOLDER) / image_name)
        anno_path = str(Path(ANNOTATION_FOLDER) / (image_name.split('.')[0] + '.xml'))
        text += f'{img_path} {anno_path}\n'
    return text.strip()


def main(): 
    train_text = create_paddle_voc_text(str(Path(args.voc_folder) / 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'))
    test_text = create_paddle_voc_text(str(Path(args.voc_folder) / 'VOCdevkit/VOC2007/ImageSets/Main/test.txt'))
    
    with open(str(Path(args.root) / 'trainval.txt'), 'w') as f:
        f.write(train_text)

    with open(str(Path(args.root) / 'test.txt'), 'w') as f:
        f.write(test_text)


parser = argparse.ArgumentParser(description="Train test split for VOC format dataset.")
parser.add_argument("--voc_folder", type=str)
parser.add_argument("--img_suffix_list", nargs='+')
parser.add_argument("--root")
args = parser.parse_args()
main()