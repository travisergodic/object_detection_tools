# PaddleYOLO

###### tags: `CV 工具`

## 環境安裝
```
$ python -m pip install paddlepaddle-gpu==2.4.1.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
$ pip install pycocotools
$ git clone https://github.com/PaddlePaddle/PaddleYOLO  # clone
$ cd PaddleYOLO
$ pip install .
```

## 資料準備
1. **COCO 格式數據集**
2. **VOC 格式數據集**


## 預測
1. **確認類別數量 & 類別個數**
   + **預設情況**：載入配置文件中 `EvalDataset` 的 `anno_path`，並產生 `clsid2catid` 與 `catid2name`。
   + **anno_path 不存在**：根據 `metric` 來載入標籤，此情況不適用自定義的數據。 
   ![](https://hackmd.io/_uploads/rk7xSqFv2.png)
   
   :::info
   詳細邏輯請見 `ppdet/data/source/category.py`
   :::
3. **執行指令**
   ```
   python -u tools/infer.py -c <config_file> -o use_gpu=true \ 
                            weights=<weight_path_or_url> \
                            --infer_img=<img_path>
   ```
   + **config_file**: 模型對應的配置文件，如 `configs/ppyoloe/objects365/ppyoloe_plus_crn_s_60e_objects365.yml`。
   + **weight_path_or_url**: 預訓練權重路徑，可為路徑或 URL。
   
     :::warning
     **PaddleYOLO** 模型配置文件中的權重檔經常沒有預訓練過，需特別注意，可以通過執行預測指令，確認模型是否成功預測，以此來確認是否使用預訓練權重。
     :::
   + **img_path**: 要預測的圖片路徑。
    
## 訓練
1. **修改配置文件**：到要訓練的模型所對應的配置文件中進行修改，修改時要特別注意以下配置
   + **資料集配置**
     + `num_classes`
     + `image_dir`
     + `anno_path`
     + `dataset_dir`
     
     ![](https://hackmd.io/_uploads/rkVjrjKv2.png)

   + **[訓練配置](https://paddledetection.readthedocs.io/tutorials/Custom_DataSet.html)**
     :::info
     1. 根据训练集数量与总 batch_size 大小计算 epoch 数，然后将epoch数换算得到训练总轮数 max_iters。milestones（学习率变化界限）也是同理。原配置文件中总 batch_size=2*8=16（8卡训练），训练集数量约为12万张，max_iters=90000，所以 epoch 数=16x90000/120000=12。在 AI 识虫数据集中，训练集数量约为 1700，在单卡GPU上训练，max_iters=12x1700/2=10200。同理计算milestones为: [6800, 9000]。
     2. 学习率与GPU数量呈线性变换关系，如果GPU数量减半，那么学习率也将减半。由于 PaddleDetection 中的 faster_rcnn_r50_fpn 模型是在 8 卡 GPU 环境下训练得到的，所以我们要将学习率除以8：
     
        ```
        max_iters: 10200
        ...
        LearningRate:
        base_lr: 0.0025
        schedulers:
        - !PiecewiseDecay
        gamma: 0.1
        milestones: [6800, 9000]
        ```
        
2. **資料格式**
   + **COCO 資料集格式**：將資料整理成下列格式，並放到 `dataset/coco` 路徑下。
     ```
     coco
     ├──annotations
     │   ├── instances_train2017.json
     │   ├── instances_val2017.json
     ├── train2017
     ├── val2017
     ```
   + **VOC 資料格式**：
     + 將資料整理成下列格式，並放到 `dataset/voc` 路徑下。
       ```
       VOCdevkit
       ├──VOC2007 (或VOC2012)
       │   ├── Annotations
       │       ├── xxx.xml
       │   ├── JPEGImages
       │       ├── xxx.jpg
       │   ├── ImageSets
       │       ├── Main
       │           ├── trainval.txt
       │           ├── test.txt
       ```
     + 產生 `tranval.txt` & `val.txt` 檔案，並放到 `dataset/voc` 路徑下，格式如下。
       ![](https://hackmd.io/_uploads/B1TGYTKP2.png)
     + 產生 `label_list.txt` 檔案，並放到 `dataset/voc` 路徑下。
       ![](https://hackmd.io/_uploads/S1qtKTtD3.png)
       :::info
       必須確保 `label_list.txt` 檔案中的類別名稱與 `xxx.xml` 檔案中的類別名稱是一致的。
       :::

3. **執行指令**
   ```
   python tools/train.py -c <config_file> --eval --amp 
   ```
   + **config_file**: 模型對應的配置文件。
   ```
   python -u tools/infer.py -c <config_file> \
              -o use_gpu=true weights=<xxx.pdparams> --infer_img=<img_path>
   ```