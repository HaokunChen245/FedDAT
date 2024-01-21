# FedDAT (Federated Dual-Adapter Teacher)

An approach for foundation model finetuning in multi-modal heterogeneous federated learning.

[Pre-print](https://arxiv.org/pdf/2308.12305.pdf)

---

## Setup

1. Create Conda environment with Python 3.8

```
conda create -n feddat python=3.8
conda activate feddat
```

2. Install requirements

```
git clone https://github.com/HaokunChen245/FedDAT.git
pip install -r requirements.txt
pip install -U adapters
pip install accelerate
```
3. Prepare datasets and pretrained-models

| Dataset | Link |
| :----:|  :----: |
| AQUA | https://github.com/noagarcia/ArtVQA/tree/master/AQUA |
| COCO-QA | http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/cocoqa-2015-05-17.zip |
| Images for COCO-QA | https://cocodataset.org/#download |
| Abstract Scenes | https://visualqa.org/download.html |
| VizWiz | https://vizwiz.org/tasks-and-datasets/vqa/ |
| GQA | https://cs.stanford.edu/people/dorarad/gqa/download.html |
| VG_100K | https://huggingface.co/datasets/visual_genome |
| Function & Scene (CLOVE benchmark) | TODO |
<!-- | Function & Scene (CLOVE benchmark) | https://github.com/showlab/CLVQA?tab=readme-ov-file | -->

Put the datasets in the folder /data

| Model | Link |
| :----:|  :----: |
| ALBEF | https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth |
| ViLT | https://huggingface.co/dandelin/vilt-b32-mlm |
| BERT | https://huggingface.co/bert-base-uncased/tree/main |

Put the models in the folder /models

---

## Run

```
# Training with ViLT
bash src/train_vilt.sh

# Training with ALBEF
bash src/train_albef.sh
```


