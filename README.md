# Project description
Image classification for early detection of diabetic retinopathy in patients. Implemented a
custom ResNet18 model from scratch using PyTorch. See the APTOS 2019 Blindness Detection competition
for the full [data overview](https://www.kaggle.com/c/aptos2019-blindness-detection/overview).

# Usage
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