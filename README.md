# TransCAM

## Prerequisite

#### 1. install dependencies 
```pip install -r requirements.txt```

#### 2. Download pretrained model
Download Conformer-S pretrained weights from https://github.com/pengzhiliang/Conformer

Download ResNet-38 pretained weights from https://github.com/YudeWang/SEAM

## Usage

### TransCAM step

#### 1. train the conformer classifier
```python train_TransCAM.py --weights {pretrained_conformer_weights}```

#### 2. TransCAM inference
```python infer_TransCAM.py --weights {trained_weights}```

#### 3. TransCAM evaluation
```python evaluation.py --predict_dir data/transcam/out_cam```

### PSA step

#### 1. train AffinityNet
```python train_aff.py```

#### 2. RW propagation
```python infer_aff.py```

#### 3. RW evaluation
```python evaluation.py --predict_dir data/transcam/out_rw --type png```

### DeepLab step

#### 1. train DeepLab
```python train_deeplab.py```

#### 2. test DeepLab
```python test_deeplab.py```

### Acknowledge 
This repo is developed based on SEAM [1] and Conformer [2]

[1] Wang, Yude, et al. "Self-supervised equivariant attention mechanism for weakly supervised semantic segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

[2] Peng, Zhiliang, et al. "Conformer: Local features coupling global representations for visual recognition." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.




