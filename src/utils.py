from pathlib import Path


def get_image_anno_pairs(img_folder, anno_folder, img_suffix_list, anno_suffix='.txt'): 
    image_anno_pair_list = []
    for image_path in Path(img_folder).glob('*'):
        if not any([image_path.name.endswith(suffix) for suffix in img_suffix_list]):
            continue
        anno_path = Path(anno_folder) / (image_path.name.split('.')[0] + anno_suffix)
        if anno_path.is_file(): 
            image_anno_pair_list.append((image_path, anno_path))

    print(f"Found {len(image_anno_pair_list)} number of image-annotation pairs.")
    return image_anno_pair_list
