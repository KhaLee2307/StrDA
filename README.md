<div align="center">
  <h1>Stratified Domain Adaptation: A Progressive Self-Training Approach for Scene Text Recognition</h1>
  <h3>WACV 2025 Early Acceptance (Round 1)</h3>
  <a href="https://openaccess.thecvf.com/content/WACV2025/html/Le_Stratified_Domain_Adaptation_A_Progressive_Self-Training_Approach_for_Scene_Text_WACV_2025_paper.html">[📰 Paper]</a>
  <a href="WACV2025/wacv25-1278-poster.pdf">[🖼️ Poster]</a>
  <a href="WACV2025/wacv25-1278-slides.pdf">[📚 Slides]</a>
  <br>
  <img src="https://wacv2025.thecvf.com/wp-content/uploads/2024/06/WACV-2025-Logo_Color-1024x315.png" width="400" alt="WACV 2025 Logo">
</div>

## News 🚀🚀🚀
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
Thanks to [ku21fan/STR-Fewer-Labels](https://github.com/ku21fan/STR-Fewer-Labels/blob/main/data.md), [baudm/parseq](https://github.com/baudm/parseq/blob/main/Datasets.md), and [Mountchicken/Union14M](https://github.com/Mountchicken/Union14M) for compiling and organizing the data. Please follow their guidelines regarding the license of each dataset.

- Evaluation datasets: LMDB archives for validation and testing
- Synthetic datasets: LMDB archives for MJSynth (MJ), SynthTex (ST)
- Real-world datasets: LMDB archives for ArT, COCO-Text (COCO), LSVT, MLT19, OpenVINO, RCTW17, ReCTS, UberText (Uber), TextOCR 

## Running the code
*Please pay attention to the warnings when running the code (e.g., select_data for target domain data, checkpoint of HDGE, and trained weights of DD).*
### Training

#### Stage 1 (Domain Stratifying)

#### Stage 2 (Progressive Self-Training)

### Testing

## Reference
If you find our work useful for your research, please cite it:
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
