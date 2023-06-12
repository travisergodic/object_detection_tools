import os
import json
import shutil
import argparse
from pathlib import Path
import sys
sys.path.insert(0, os.getcwd())

import funcy
import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

TRAIN_IMAGE_FOLDER = 'coco/train2017'
TEST_IMAGE_FOLDER = 'coco/val2017'
ANNOTATION_FOLDER = 'coco/annotations'


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def filter_images(images, annotations):

    annotation_ids = funcy.lmap(lambda i: int(i['image_id']), annotations)

    return funcy.lfilter(lambda a: int(a['id']) in annotation_ids, images)


parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('-s', dest='split', type=float, required=True,
                    help="A percentage of a split; a number in (0, 1)")
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')

parser.add_argument('--multi-class', dest='multi_class', action='store_true',
                    help='Split a multi-class dataset while preserving class distributions in train and test sets')
parser.add_argument('--img_folder')
parser.add_argument('--root')

args = parser.parse_args()

def main(args):
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco.get('info')
        licenses = coco.get('licenses')
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        (Path(args.root) / TRAIN_IMAGE_FOLDER).mkdir(parents=True, exist_ok=False)
        (Path(args.root) / TEST_IMAGE_FOLDER).mkdir(parents=True, exist_ok=False)
        (Path(args.root) / ANNOTATION_FOLDER).mkdir(parents=True, exist_ok=False)

        if args.multi_class:

            annotation_categories = funcy.lmap(lambda a: int(a['category_id']), annotations)

            #bottle neck 1
            #remove classes that has only one sample, because it can't be split into the training and testing sets
            annotation_categories =  funcy.lremove(lambda i: annotation_categories.count(i) <=1  , annotation_categories)

            annotations =  funcy.lremove(lambda i: i['category_id'] not in annotation_categories  , annotations)


            X_train, y_train, X_test, y_test = iterative_train_test_split(np.array([annotations]).T,np.array([ annotation_categories]).T, test_size = 1-args.split)

            save_coco(str(Path(args.root) / ANNOTATION_FOLDER / 'train2017.json'), info, licenses, filter_images(images, X_train.reshape(-1)), X_train.reshape(-1).tolist(), categories)
            save_coco(str(Path(args.root) / ANNOTATION_FOLDER / 'test2017.json'), info, licenses,  filter_images(images, X_test.reshape(-1)), X_test.reshape(-1).tolist(), categories)

            print("Saved {} entries in {} and {} in {}".format(len(X_train), TRAIN_IMAGE_FOLDER, len(X_test), TEST_IMAGE_FOLDER))

            # copy image to dst folder
            for filename in [image['file_name'] for image in filter_images(images, X_train.reshape(-1))]:
                shutil.copy(str(Path(args.img_folder) / filename), str(Path(args.root) / TRAIN_IMAGE_FOLDER / filename)) 
            
            for filename in [image['file_name'] for image in filter_images(images, X_test.reshape(-1))]:
                shutil.copy(str(Path(args.img_folder) / filename), str(Path(args.root) / TEST_IMAGE_FOLDER / filename))

        else:

            X_train, X_test = train_test_split(images, train_size=args.split)

            anns_train = filter_annotations(annotations, X_train)
            anns_test=filter_annotations(annotations, X_test)

            save_coco(str(Path(args.root) / ANNOTATION_FOLDER / 'train2017.json'), info, licenses, X_train, anns_train, categories)
            save_coco(str(Path(args.root) / ANNOTATION_FOLDER / 'test2017.json'), info, licenses, X_test, anns_test, categories)

            print("Saved {} entries in {} and {} in {}".format(len(anns_train), TRAIN_IMAGE_FOLDER, len(anns_test), TEST_IMAGE_FOLDER))
            
            # copy images to dst folder
            for filename in X_train:
                shutil.copy(str(Path(args.img_folder) / filename), str(Path(args.root) / TRAIN_IMAGE_FOLDER / filename)) 
            
            for filename in X_test:
                shutil.copy(str(Path(args.img_folder) / filename), str(Path(args.root) / TEST_IMAGE_FOLDER / filename))


if __name__ == "__main__":
    main(args)