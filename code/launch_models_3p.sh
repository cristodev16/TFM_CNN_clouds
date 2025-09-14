#!/usr/bin/bash

echo "Launching pretrained ResNet34 training and testing..."
nohup python3 -u train_test_model.py -m "resnet34" -twvd -s -cw -lr 0.0001 > log.out 2>&1 &
sleep 20
latest_dir=$(find ../data/ -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
mv log.out "$latest_dir/log.out"

#echo "Launching pretrained DenseNet121 training and testing..."
#nohup python3 -u train_test_model.py -m "densenet121" -p -twvd -s -cw -lr 0.0001> log.out 2>&1 &
#sleep 20
#latest_dir=$(find ../data/ -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
#mv log.out "$latest_dir/log.out"

#echo "Launching pretrained VGG16 training and testing..."
#nohup python3 -u train_test_model.py -m "vgg16" -twvd -s -cw -lr 0.0001 > log.out 2>&1 &
#sleep 20 
#latest_dir=$(find ../data/ -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
#mv log.out "$latest_dir/log.out"
