# Data Source
Data for this project is obtained from the 
[APTOS 2019 Blindness Detection competition](https://www.kaggle.com/c/aptos2019-blindness-detection/data) on Kaggle.

To download the data using [Kaggle API](https://github.com/Kaggle/kaggle-api/blob/master/README.md):
```bash
kaggle competitions download -c aptos2019-blindness-detection
```
Training and test data is by default expected in `data` directory. Run `python train.py -h` for 
expected parameters.

## Data Description
You are provided with a large set of retina images taken using fundus photography under a variety of imaging conditions.

A clinician has rated each image for the severity of diabetic retinopathy on a scale of 0 to 4:

0 - No DR

1 - Mild

2 - Moderate

3 - Severe

4 - Proliferative DR

Like any real-world data set, you will encounter noise in both the images and labels. Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

