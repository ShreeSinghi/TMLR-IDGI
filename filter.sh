#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --output=output1.log
#SBATCH --time=24:00:00  # Change the time limit to 72 hours
#SBATCH --job-name=1script

#python script.py --model=mobilenetv2 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128   1  128
#python script.py --model=resnet50v2 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128    1  128
#python script.py --model=resnet101v2 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128   2  96
#python script.py --model=xception --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128      fuckit  32  
#python script.py --model=inceptionv3 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128   3  128
#python script.py --model=vgg16 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128         hogaya  128 
#python script.py --model=vgg19 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128         1  96
#python script.py --model=densenet121 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128   4  96 
#python script.py --model=densenet169 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128   3  64
#python script.py --model=densenet201 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128   2  64
#python script.py --model=resnet152v2 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128   4  64

conda activate mlrc
python script.py --model=mobilenetv2 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128 --method=standard --n_steps=128 --class_cluster=1 
python script.py --model=resnet50v2 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128 --method=standard --n_steps=128 --class_cluster=1
python script.py --model=vgg19 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=96 --method=standard --n_steps=96 --class_cluster=1 --resume_epoch=940