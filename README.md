# 物件偵測工具

## 標籤轉換
1. **YOLOv5 轉 COCO**
   ```python
   yolo = AnnotationSet.from_yolo_v5(
        folder="path/to/label/dir",
        image_folder="path/to/image/dir"
   )

   yolo.save_coco('path/to/coco/json/file', auto_ids=True)
   ```

2. **YOLOv5 轉 VOC**
   ```python
   yolo = AnnotationSet.from_yolo_v5(
        folder="path/to/label/dir",
        image_folder="path/to/image/dir"
   )

   yolo.save_pascal_voc('path/to/xml/file/dir')
   ```

## 資料分割
1. **分割 COCO 數據集**
   ```
   python cocosplit.py --annotations <annotations> \
                       --split <split_ratio> \
                       --having-annotations <having_annotations> \
                       --multi-class <multiclass> \ 
                       --img_folder <img_folder> \
                       --root <root>
   ```
   + **annotations**: Path to COCO annotations file.
   + **split**: A percentage of a split; a number in (0, 1).
   + **having_annotations**: Ignore all images without annotations. Keep only these with at least one annotation.
   + **multiclass**: Split a multi-class dataset while preserving class distributions in train and test sets. 
   + **img_folder**: Path to image folder.
   + **root**: directory for split coco folder.

   ```
   coco
   ├──annotations
   │   ├── instances_train2017.json
   │   ├── instances_val2017.json
   ├── train2017
   ├── val2017
   ```

2. **分割 YOLO 數據集**
   ```
   python yolosplit.py --img_foler <img_folder> \
                       --anno_folder <anno_folder> \
                       --img_suffix_list <img_suffix_list> \
                       --train_ratio <train_ratio> \
                       --root <root>
   ```
   + **img_folder**: Path to image folder.
   + **anno_folder**: Path to yolo annotation folder.
   + **img_suffix_list**: Image suffixs.
   + **train_ratio**: A percentage of a split; a number in (0, 1).
   + **root**: Directory for split yolo folder.

   ```
   train
   ├──images
   │   ├── xxx.<ext>
   │   ├── xxx.<ext>
   │   ├── ...
   ├── labels
   │   ├── xxx.txt
   │   ├── xxx.txt 
   │   ├── ...
   val
   ├──images
   │   ├── xxx.<ext>
   │   ├── xxx.<ext>
   │   ├── ...
   ├── labels
   │   ├── xxx.txt
   │   ├── xxx.txt 
   │   ├── ...
   ```

3. **分割 VOC 數據集**
   ```
   python vocsplit.py --img_folder <img_folder> \
                      --anno_folder <anno_folder> \
                      --img_suffix_list <img_suffix_list> \
                      --train_ratio <train_ratio> \
                      --root <root>
   ```
   + **img_folder**: Path to image folder.
   + **anno_folder**: Path to xml annotation folder.
   + **img_suffix_list**: image suffixs.
   + **train_ratio**: A percentage of a split; a number in (0, 1).
   + **root**: Directory for split yolo folder.

   ```
   VOC2007
   ├── Annotations
   │   ├── xxx.xml
   │   ├── xxx.xml
   │   ├── ...
   ├── JPEGImages
   │   ├── xxx.<ext>
   │   ├── xxx.<ext>
   │   ├── ...
   ├── ImageSets
   │   ├── Main
   │   │   ├── trainval.txt
   │   │   ├── test.txt
   ```
