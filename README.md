# Project Description
Image classification for early detection of diabetic retinopathy in patients. Classification is 
perfomed on retina images of patients taken using fundus photography. This project uses a
custom ResNet18 model built from scratch using PyTorch. 

> Disclaimer: Not intended for medical diagnosis. This project analyzes medical images for demonstration 
> purposes only. Always consult with your doctor, or other qualified healthcare professional before
> self diagnosing.

# Data Source
See the APTOS 2019 Blindness Detection competition
for the full [overview](https://www.kaggle.com/c/aptos2019-blindness-detection/overview) and [data
description](https://www.kaggle.com/c/aptos2019-blindness-detection/data) on Kaggle.

To download the data using [Kaggle API](https://github.com/Kaggle/kaggle-api/blob/master/README.md):
```bash
kaggle competitions download -c aptos2019-blindness-detection
```
Training and test data is by default expected in `data` directory. Run `python train.py -h` or `python infer.py -h` for 
expected parameters.

# Usage
Clone repository:
```shell
git clone https://github.com/thatgeeman/retinopathy_classification_resnet18
```
Setup environment and install dependencies:
```shell
pip install pipenv
cd retinopathy_classification_resnet18
pipenv install
pipenv shell
```

To train the model from the data in `data/train.csv` with 
images located in `data/train_images`
```shell
python train.py 2 10 --csv data/train.csv --data data/train_images
```
Here, the first parameter denotes the number of epochs to train the model with
frozen body parameters. The second parameter denotes the number of epochs to train
the full model.

Using the saved checkpoint to run an inference cycle:
```shell
python infer.py checkpoints/model_c15.pth
```
