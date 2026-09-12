# SAR Object Detection — SARDet-100K Benchmark

This project benchmarks YOLOv12 on the SARDet-100K dataset, a large-scale synthetic aperture radar (SAR) object detection benchmark. The goal is to evaluate YOLOv12's detection performance against the MSFA baseline proposed in the NeurIPS 2024 paper, exploring how modern real-time detection architectures adapt to the unique characteristics of SAR imagery, including speckle noise, target scattering signatures, and the absence of optical color features.

## Dataset

SARDet-100K — a large-scale, multi-class SAR object detection dataset.

Repository: https://github.com/zcablii/SARDet_100K

> **Note:** The `data/` folder is excluded from this repository. Download the dataset separately and place it under `data/`.

## Model

YOLOv12 — attention-centric real-time object detector.

Repository: https://github.com/sunsmarterjie/yolov12

## Results

**YOLOv12-n** was trained in three sequential stages of 50 epochs each. Each stage resumed from the previous stage's final weights with a lower initial learning rate. All metrics are on the SARDet-100K **validation** split (10,492 images), image size 640.
 
| Stage | Epochs (cumulative) | Notes                                                    | mAP@50    | mAP@50–95 |
| ----- | ------------------- | -------------------------------------------------------- | --------- | --------- |
| Run 1 | 50                  | lr0 = 0.01; 18 epochs on Kaggle 2× T4, then Colab A100   | 0.825     | 0.528     |
| Run 2 | 100                 | resumed from Run 1, lower lr0                            | 0.866     | 0.571     |
| Run 3 | 150                 | resumed from Run 2, lower lr0; ~6 h                      | **0.881** | **0.591** |
 
Run 2's gains came mostly from the weaker classes, tank AP@50 rose from 0.622 to 0.686 and bridge from 0.578 to 0.683. Run 3 improved only marginally, suggesting the nano model is approaching its capacity on this dataset.
 
**Against the SARDet-100K paper.** At 2.6 M parameters, YOLOv12-n reaches 59.1 mAP@50–95, which sits above the ResNet / ConvNeXt / VAN / Swin backbones in the paper's Fig. 4(b) (roughly 47–56 mAP@50–95 at 3–90 M parameters). One caveat: our numbers are validation-set figures, while the paper's are its reported benchmark results, so the two are not measured under an identical protocol. Read the comparison as indicative rather than like-for-like.
 
**Demo check.** Before the full runs, we did a 1-epoch dry run on 10 % of the training set at 320 px on an RTX 3060 (WSL) gave mAP@50 = 0.048. expected for one epoch, and enough to confirm the data pipeline end to end (`notebooks/YOLOtest.ipynb`).

## Results write up 


