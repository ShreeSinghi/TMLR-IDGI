#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/output2.log
#SBATCH --time=24:00:00
#SBATCH --job-name=2script


# python proba_filterer.py --model=mobilenetv2 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128  done
# python proba_filterer.py --model=resnet50v2 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128   done
# python proba_filterer.py --model=inceptionv3 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128  done
# python proba_filterer.py --model=vgg16 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=128        done 
# python proba_filterer.py --model=resnet101v2 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=96   done
# python proba_filterer.py --model=vgg19 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=96         done
# python proba_filterer.py --model=densenet121 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=96   done
# python proba_filterer.py --model=densenet169 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=64   done
# python proba_filterer.py --model=densenet201 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=64   ,2 pic
# python proba_filterer.py --model=resnet152v2 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=64   4  insertion

python script_pic.py --model=densenet201 --dataroot=/scratch/shree_s.iitr/imagenet --batchSize=64 --method=standard --n_steps=128
