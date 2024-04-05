#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/output2.log
#SBATCH --time=24:00:00  # Change the time limit to 72 hours
#SBATCH --job-name=2script


# python proba_filterer.py --model=mobilenetv2 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128
# python proba_filterer.py --model=resnet50v2 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128
# python proba_filterer.py --model=inceptionv3 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128 
# python proba_filterer.py --model=vgg16 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128        
# python proba_filterer.py --model=resnet101v2 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=96 
# python proba_filterer.py --model=vgg19 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=96        
# python proba_filterer.py --model=densenet121 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=96  
# python proba_filterer.py --model=densenet169 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=64   
# python proba_filterer.py --model=densenet201 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=64  
# python proba_filterer.py --model=resnet152v2 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=64   

conda activate mlrc
python script.py --model=inceptionv3 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128 --method=standard --n_steps=128 --class_cluster=1