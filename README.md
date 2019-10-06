# Style Augmentation
This code is an implementation of the paper [Style Augmentation: Data Augmentation via Style Randomization
](https://arxiv.org/abs/1809.05375). My main contributions here are the following:

- We used Xception and InceptionV4 as classifier networks.
- We used [Learning Linear Transformations for Fast Image and Video Style Transfer](https://github.com/sunshineatnoon/LinearStyleTransfer) as stylizing network.
- We trained the styliying net using the r31 and saving the embedded features in the same manner as the original style tranfer paper (however, pretrained methods are available for the stylizing network.


## Personal Targets
- Unterstand Why texture bias the training as stated in [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://arxiv.org/abs/1811.12231)
- Understand the influence between stylization in augmentation and networks that perform this task (I tested and read several style transfer, but this one was the best).
- Train networks and use a network as data augmenter embedded in torchvision.
- Understand the incluence of data augmentation in each network and hyperparameter fitting
- Analyze the noise in the latent domain (uniform and gaussian were tested).

## Usage
All parameters are detailed in `main.py` and `config_classification.py`, just download the [wikiArt (training) DB](https://www.kaggle.com/c/painter-by-numbers/data), the pretrained The model for stylization from [original authors](https://drive.google.com/file/d/1H9T5rfXGlGCUh04DGkpkMFbVnmscJAbs/view) or you can follow the author's procedure to get the model, thereafter move the model into models folder. Finally, run the training code.:


```
python FeatureExtractor.py # To get the feature vectors from WikiArt DB.
python training.py
```
hyperparameters can be edited in the `config_classification.py`

### Results

Results reported on Xception and InceptionV4 using STL-10 for classfication task are the following: 

| Netwrok    |    Trad   |   Style   |  Accuracy  | 
| ---------  | --------- | --------- | ---------- |
| Xception   |           |           |   68.17%   |
| Xception   |     X     |           |   83.65%   |
| Xception   |           |     X     |   69.28%   |
| Xception   |     X     |     X     |      -     |
| InceptionV4|           |           |   74.08%   |
| InceptionV4|     X     |           |   86.06%   |
| InceptionV4|           |     X     |   75.69%   |
| InceptionV4|     X     |     X     |      -     |

Weights tuned without pre-training are available in the following: [dropbox](https://www.dropbox.com/sh/pui7hla90zvqbjk/AABRxVS8xGjhTbiswi1dTCyVa?dl=0)

Some qualitative results for style transfer.

No Augmentation (original images)
![No Augmentation](test/test_augmentation_nothing.png)
Trad+Style augmentation ( ColorJitter, RandomHorizontalFlip, RandomAffine, RandomRotation, RandomErasing)
![Trad+Style augmentation](test/test_augmentation_aug.png)





