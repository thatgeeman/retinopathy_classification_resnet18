# Project Description
Image classification for early detection of diabetic retinopathy in patients. Implemented a
custom ResNet18 model from scratch using PyTorch. See the APTOS 2019 Blindness Detection competition
for the full [data overview](https://www.kaggle.com/c/aptos2019-blindness-detection/overview).

# Data Source
Data for this project is obtained from the 
[APTOS 2019 Blindness Detection competition](https://www.kaggle.com/c/aptos2019-blindness-detection/data) on Kaggle.

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