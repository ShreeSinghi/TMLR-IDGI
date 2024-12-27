## Strengthening Interpretability: An Investigative Study of Integrated Gradient Methods

This repository contains the official implementation of the following paper accepted at [TMLR](https://jmlr.org/tmlr/):

> **Strengthening Interpretability: An Investigative Study of Integrated Gradient Methods**<br>
> Shree Singhi, Anupriya Kumari <br>
> https://www.arxiv.org/abs/2409.09043
>
> **Abstract:** *We conducted a reproducibility study on Integrated Gradients (IG) based methods and the Important Direction Gradient Integration (IDGI) framework. IDGI eliminates the explanation noise in each step of the computation of IG-based methods that use the Riemann Integration for integrated gradient computation. We perform a rigorous theoretical analysis of IDGI and raise a few critical questions that we later address through our study. We also experimentally verify the authors' claims concerning the performance of IDGI over IG-based methods. Additionally, we varied the number of steps used in the Riemann approximation, an essential parameter in all IG methods, and analyzed the corresponding change in results. We also studied the numerical instability of the attribution methods to check the consistency of the saliency maps produced. We developed the complete code to implement IDGI over the baseline IG methods and evaluated them using three metrics since the available code was insufficient for this study.*

### Usage: 

Create a folder `<dataroot>/val/images` and download and place all 50K validation images of ImageNet

Copy and paste `val_annotations.txt` inside `<dataroot>/val/`

Run `lossy.py` and `blur.py` to generate compressed and blurred images for metric calculations.

Note that for each model and metric the saliencies and results of the baseline method and baseline method+IDGI are calculated together. Each set of saliencies takes ~30GB of space
For each model and each attribution method the results need to be evaluated on, do the following:
1. Run `class_filterer.py` to filter out correctly predicted images for a model
2. Run `proba_filterer.py` to filter out correctly predicted images with confidence >= 80% for a model
3. Run `script.py` to generate saliencies for a model for a chosen attribution method
4. Run `script_lossy.py` to get the MSE between the original image and compressed image saliencies on the terminal
5. Run `script_insertion.py` to get the insertion scores in `./results/`
6. Run `script_pic.py` to get the AUC AIC and AUC SIC scores in `./results/`

#### Authors: 
[Shree Singhi](https://github.com/ShreeSinghi), [Anupriya Kumari](https://github.com/anupriyakkumari)
