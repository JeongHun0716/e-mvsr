# Efficient-Training-for-Multilingual-Visual-Speech-Recognition

This repository contains the PyTorch implementation of the following paper:
> **Efficient-Training-for-Multilingual-Visual-Speech-Recognition: Pre-training with Discretized Visual Speech Representation**<be>
><br>
>**(ACM MM 2024)**<br>
> \*Minsu Kim, \*Jeonghun Yeo, Se Jin Park, Hyeongseop Rha, Yong Man Ro<br>
> \[[Paper](https://openreview.net/forum?id=rD7guYi6jZ)\]


<div align="center"><img width="100%" src="img/img.png?raw=true" /></div>


## Environment Setup
```bash
conda create -n e-mvsr python=3.9 -y
conda activate e-mvsr
git clone https://github.com/JeongHun0716/e-mvsr
cd e-mvsr
```

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
(If your pip version > 24.1, please run "python3 -m pip install --upgrade pip==24.0")
pip install numpy==1.23.5 editdistance omegaconf==2.0.6 hydra-core==1.0.7
pip install python_speech_features scipy opencv-python einops soundfile
pip install sentencepiece tqdm tensorboard
```

## Dataset preparation
For inference, Multilingual TEDx(mTEDx), and LRS3 Datasets are needed. 
  1. Download the mTEDx dataset from the [mTEDx link](https://www.openslr.org/100) of the official website.
  2. Download the LRS3 dataset from the [LRS3 link](https://mmai.io/datasets/lip_reading/) of the official website.
     
When you are interested in training the model, you should prepare additional VoxCeleb2 and AVSpeech datasets. 

  3. Download the VoxCeleb2 dataset from the [VoxCeleb2 link](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) of the official website.
  4. Download the AVSpeech dataset from the [AVSpeech link](https://looking-to-listen.github.io/avspeech/) of the official website.

## Preprocessing 
After downloading the datasets, you should detect the facial landmarks of all videos and crop the mouth region using these facial landmarks. We recommend you preprocess the videos following [Visual Speech Recognition for Multiple Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages).  

  
## Inference
Download the checkpoints from the below links and move them to the `src/pretrained_models` directory. 
You can evaluate the performance of the finetuned model using the scripts available in the `scripts` directory.

## Pretrained Models


| Model         | Training Datasets  | Training data (h)  |    Used Languages     |
|--------------|:----------|:------------------:|:----------:|
| [mavhubert.pt](https://www.dropbox.com/scl/fo/pv9ldbregra0okmykfu3f/AP53yir_loStodqzl6UD0a0?rlkey=ik0jdy6s1fdr62iuzyn6eopf6&st=pjk2qhnp&dl=0) |       mTEDx + LRS3 + VoxCeleb2 + AVSpeech + LRS2       |        5,512           |     En, Es, It, Fr, Pt, De, Ru, Ar, El  |
| [unit_pretrained.pt](https://www.dropbox.com/scl/fo/ar81hqaf7t3bkx3b326cu/AAYOFhduVMLDFPwfzJhnLpE?rlkey=ku7b9209w9rljumenibqjg2v0&st=vl4kgkn4&dl=0) |        mTEDx + LRS3 + VoxCeleb2 + AVSpeech            |        4,545          |     En, Es, It, Fr, Pt  |
| [finetuned.pt](https://www.dropbox.com/scl/fo/s99l48686q0lrpcugjvi1/AAptetzX0TRr5YPQwRlXOLc?rlkey=as88xz20ojtwof1tizvmwzl8c&st=trb74thj&dl=0) |       mTEDx + LRS3 + VoxCeleb2 + AVSpeech     |        4,545         |    En, Es, It, Fr, Pt  |

## Citation
If you find this work useful in your research, please cite the paper:

```bibtex
@inproceedings{kim2024efficient,
  title={Efficient Training for Multilingual Visual Speech Recognition: Pre-training with Discretized Visual Speech Representation},
  author={Kim, Minsu and Yeo, Jeonghun and Park, Se Jin and Rha, Hyeongseop and Ro, Yong Man},
  booktitle={ACM Multimedia 2024}
}
```


## Acknowledgement

This project is based on the [avhubert](https://github.com/facebookresearch/av_hubert) and [fairseq](https://github.com/facebookresearch/fairseq) code. We would like to thank the developers of these projects for their contributions and the open-source community for making this work possible.
