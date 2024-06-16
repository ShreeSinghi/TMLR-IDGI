Follow the following steps in order to completely reproduce results of integrated gradient methods:

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
