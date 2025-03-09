<div align="center">
  <h1>Stratified Domain Adaptation: A Progressive Self-Training Approach for Scene Text Recognition</h1>
  <a href="https://openaccess.thecvf.com/content/WACV2025/html/Le_Stratified_Domain_Adaptation_A_Progressive_Self-Training_Approach_for_Scene_Text_WACV_2025_paper.html">[📰 Paper]</a>
  <a href="WACV2025/wacv25-1278-poster.pdf">[🖼️ Poster]</a>
  <a href="WACV2025/wacv25-1278-slides.pdf">[📚 Slides]</a>
  <br>
  <h3>WACV 2025 Early Acceptance (Round 1)</h3>
  <img src="https://wacv2025.thecvf.com/wp-content/uploads/2024/06/WACV-2025-Logo_Color-1024x315.png" width="400" alt="WACV 2025 Logo">
</div>

## Introduction
This is the official PyTorch implementation of the [StrDA paper](https://openaccess.thecvf.com/content/WACV2025/html/Le_Stratified_Domain_Adaptation_A_Progressive_Self-Training_Approach_for_Scene_Text_WACV_2025_paper.html), which was accepted at the main conference of the ***IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025***.

In this paper, we propose the Stratified Domain Adaptation (StrDA) approach, a progressive self-training framework for scene text recognition. By leveraging the gradual escalation of the domain gap with the Harmonic Domain Gap Estimator ($\mathrm{HDGE}$), we propose partitioning the target domain into a sequence of ordered subsets to progressively reduce the domain gap between each and the source domain. Progressive self-training is then applied sequentially to these subsets. Extensive experiments on STR benchmarks demonstrate that our approach enables the baseline STR models to progressively adapt to the target domain. This approach significantly improves the performance of the baseline model without using any human-annotated data and shows its superior effectiveness compared to existing UDA methods for the scene text recognition task.

* **Keywords:** scene text recognition (STR), unsupervised domain adaptation (UDA), self-training (ST), optical character recognition (OCR)

## News 🚀🚀🚀
- `2025/03/06`: 📜 We have uploaded the instructions for running the code.
- `2025/03/03`: 💻 We have released the implementation of StrDA for TRBA and CRNN.
- `2025/02/28`: 🗣️ We attended the conference, you can view the poster and slides [here](WACV2025).
- `2025/08/30`: 🔥 Our paper has been accepted to [WACV'25](https://wacv2025.thecvf.com/) (Algorithms Track).

## Getting Started

### Installation
1. python>=3.8.16
2. Install PyTorch-cuda>=11.3 following [official instruction](https://pytorch.org/):

        pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
        
3. Install the necessary dependencies by running (`!pip install -r requirements.txt`):

        pip install opencv-python==4.4.0.46 Pillow==7.2.0 opencv-python-headless==4.5.1.48 lmdb tqdm nltk six pyyaml

* You can also create the environment using `docker build -t StrDA .`

### Datasets
Thanks to [ku21fan/STR-Fewer-Labels](https://github.com/ku21fan/STR-Fewer-Labels/blob/main/data.md), [baudm/parseq](https://github.com/baudm/parseq/blob/main/Datasets.md), and [Mountchicken/Union14M](https://github.com/Mountchicken/Union14M) for compiling and organizing the data. I highly recommend that you follow their guidelines to download the datasets and review the license of each dataset.

## Running the code
*Please pay attention to the warnings when running the code (e.g., select_data for target domain data, checkpoint of HDGE, and trained weights of DD).*

### Training

- First, you need a **source-trained STR model**. If you don’t have one, you can use `supervised_learning.py` to train an STR model with **source domain data (synthetic)**.
- Next, you need to **filter the data**, removing samples that are too long (width > 25 times height) and save them to `select_data.npy` (to be updated later). Since the model only processes a maximum of 25 characters per word, these long samples could be **harmful** during pseudo-labeling.
- Then, you will **run Stage 1** using one of the two methods. The files containing data information for each subset will be saved in `stratify/{args.method}/` as `.npy` files. **Please check them carefully!** 
- Finally, **run Stage 2** to perform adaptation on the **target domain data** to **boost model performance**. Then, test the results using a wide range of benchmarks.

*Note: The target domain data must remain **unchanged** throughout the experiment.*

#### Supervised Learning

    CUDA_VISIBLE_DEVICES=0 python supervised_learning.py --model TRBA --aug

#### Stage 1 (Domain Stratifying)

There are 2 main methods with many settings:
1. Harmonic Domain Gap Estimator ($\mathrm{HDGE}$)

        CUDA_VISIBLE_DEVICES=0 python stage1_HDGE.py --select_data select_data.npy --num_subsets 5 --beta 0.7 --train

2. Domain Discriminator ($\mathrm{DD}$)

        CUDA_VISIBLE_DEVICES=0 python stage1_DD.py --select_data select_data.npy --num_subsets 5 --discriminator CRNN --train --aug

*Note: For both methods, you only need to activate `--train` to train the model the first time. After that, you can stratify the data without retraining.*

#### Stage 2 (Progressive Self-Training)

    CUDA_VISIBLE_DEVICES=0 python stage2_StrDA.py --saved_model trained_model/TRBA.pth --model TRBA --num_subsets 5 --method HDGE --beta 0.7 --aug

*Note: If the method is HDGE, you must enter `--beta`. If the method is DD, you must select a `--discriminator`. Example:*

    CUDA_VISIBLE_DEVICES=0 python stage2_StrDA.py --saved_model trained_model/CRNN.pth --model CRNN --num_subsets 5 --method DD --discriminator CRNN --aug

### Testing

    CUDA_VISIBLE_DEVICES=0 python test.py --saved_model trained_model/TRBA.pth --model TRBA

**Broader insight:** You can try this method with different STR models, on various source-target domain pairs (e.g., synthetic-handwritten/art text) and even more complex domain gap problems like medical image segmentation. Additionally, you can replace self-training with more advanced UDA techniques.

## Reference
If you find our work useful for your research, please cite it and give us a star⭐!
```
@InProceedings{Le_2025_WACV,
    author    = {Le, Kha Nhat and Nguyen, Hoang-Tuan and Tran, Hung Tien and Ngo, Thanh Duc},
    title     = {Stratified Domain Adaptation: A Progressive Self-Training Approach for Scene Text Recognition},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {8972-8982}
}
```

## Acknowledgements
This code is based on [STR-Fewer-Labels](https://github.com/ku21fan/STR-Fewer-Labels) by [Jeonghun Baek](https://github.com/ku21fan) and [cycleGAN-PyTorch](https://github.com/arnab39/cycleGAN-PyTorch) by [Arnab Mondal
](https://github.com/arnab39). Thanks for your contributions!
