# Dlchemist


This repository presents a comprehensive collection of jupyter notbooks that replicates the original results of many deep learning models or papers. Each implementation is built from scratch using Pytorch and run on Google Colab, using either the original datasets or a subset of them.


## Image Classification

| Network | Dataset | Preprocess | Inference | Accuracy | Status |
| --------| --------| -----------|----------| ------ | ---- |
|  [LeNet-5](https://github.com/kevinkevin556/Dlchemist/blob/main/lenet5.ipynb) | Mnist |  | | 98.70 % | ![Writing](https://img.shields.io/static/v1.svg?label=Status&message=Writing&color=yellow)
| [AlexNet](https://github.com/kevinkevin556/Dlchemist/blob/main/alexnet.ipynb) | Cifar-10 |  PCA color jittering| | 84.99 % | ![Writing](https://img.shields.io/static/v1.svg?label=Status&message=Writing&color=yellow)
|[Network in Network](https://github.com/kevinkevin556/Dlchemist/blob/main/nin2.ipynb) | Cifar-10 | ZCA, GCN | | 88.29 % | ![Writing](https://img.shields.io/static/v1.svg?label=Status&message=Writing&color=yellow)
| VGG | Imagenette | normalize, random crop,horizontal flip | FC to Conv, Average softmax (2)| 88.53 % | ![Writing](https://img.shields.io/static/v1.svg?label=Status&message=Writing&color=yellow)
| GoogleNet (Inception-v1) | Imagenette | normalize, random crop, photometric distortions | Polyak averaging, Average softmax (36) | 81.57 % | ![Building](https://img.shields.io/static/v1.svg?label=Status&message=Building&color=red)



<dir style="display:None">

status: 
![Building](https://img.shields.io/static/v1.svg?label=Status&message=Building&color=red)

![Writing](https://img.shields.io/static/v1.svg?label=Status&message=Writing&color=yellow)

![Finished](https://img.shields.io/static/v1.svg?label=Status&message=Finished&color=green)

</dir>