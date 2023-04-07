#!/bin/bash

# Prompt the user to enter hyperparameters
read -p "Enter batch size: " batch_size
read -p "Enter number of epochs: " epochs
read -p "Enter initial learning rate: " lr_init
read -p "Enter weight decay parameter: " decay
read -p "Enter momentum parameter: " momentum

# Run the Python script with the specified hyperparameters
python ./srcs/builder.py --dir='cifar-10/cifar-10-bathces-py' --batch_size=$batch_size --epochs=$epochs --lr_init=$lr_init --wd=$decay --momentum=$momentum