#!/bin/bash

python main.py -i 1 -n 100 -m test -s test -l False -c 1 -p ./save/ -f outputs/log.txt -a 9 -e 2 -o 1 -r 0.001 -x True
# -m for the mode of "train" or "test"
# -s for the set  of "train", "test" or "valid"
# -l for using the stored models or not (-l True)
# -c to specify the gpu number
# -p to specify the main folder for the experiment ( Save and load)
# -f to specify the txt file path and name for saving log
# -a for architecture number (7,8,9)
# -e specifying the embedding type ( 1 for bert, 2 for flair, 3 for xlnet)
# -o For specifying the loss mode ("one" for objective 1 and "all" for objective 2)
# -r for specifying the learning rate
# -x for enabling or desabling the modified max pooling
