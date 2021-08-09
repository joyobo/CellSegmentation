# CellSegmentation
50.021: AI -- Cell Segmentation 

## Dataset 
Colorectal Nuclear Segmentation and Phenotypes (CoNSeP) Dataset
S. Graham, Q. D. Vu, S. E. A. Raza, A. Azam, Y-W. Tsang, J. T. Kwak and N. Rajpoot. "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images." Medical Image Analysis, Sept. 2019. https://doi.org/10.1016/j.media.2019.101563

## Introduction 
While many approaches and algorithms have been explored by researchers in the field of cell segmentation, HoVer-Net (Graham et al., 2019) has achieved state-of-art performance as compared to the other published approaches. As such, the project aims to find an alternative model that performs the best in segmenting cells from the dataset proposed by the HoVer-Net using CNN. 

## Pre-processing Dataset (Data Augmentation) -- Description of notebook
`Five Crop.ipynb` crops the input image and its corresponding label into 5 parts -- the four corners and the center.

## Experiments -- Description of notebooks
Following notebooks are our experiments to find the best model and our hyperparameter tuning on the best model (ResNet-18 Residual Attention U-Net) to find the best frame size, batch size and learning rate.
1. `Training-ResNet18AttentionUNet.ipynb` :  Trained the ResNet-18 Residual Attention U-Net model with 500x500 frame size, batch size 1 and 0.001 learning rate and hyperparameter tuned with varying frame sizes, batch sizes and learning rates
2. `Training-CustomAttentionUNet.ipynb` : Trained the Custom Residual Attention U-Net model with 500x500 frame size, batch size 1 and 0.001 learning rate
3. `Training_EfficientNetAttentionUNet.ipynb` : Trained the EfficientNet-B7 Residual Attention U-Net model with 500x500 frame size, batch size 1 and 0.001 learning rate<br><br>  

## Testing -- Description of notebook
A saved model can be loaded in `Testing_ResNet18AttentionUNet.ipynb` notebook to test the ResNet-18 Residual Attention U-Net model on test set.

## GUI
Read details at `./GUI/README.md`.
