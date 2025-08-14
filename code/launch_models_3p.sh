#!/usr/bin/bash

echo "Launching pretrained ResNet18 training and testing..."
nohup python3 -u train_test_model.py -m "resnet18" -p -r -ft > log.out 2>&1 &
sleep 15
latest_dir=$(find ../data/ -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
mv log.out "$latest_dir/log.out"

echo "Launching pretrained ResNet34 training and testing..."
nohup python3 -u train_test_model.py -m "resnet34" -p -r -ft > log.out 2>&1 &
sleep 15 
latest_dir=$(find ../data/ -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
mv log.out "$latest_dir/log.out"

echo "Launching pretrained ResNet50 training and testing..."
nohup python3 -u train_test_model.py -m "resnet50" -p -r -ft > log.out 2>&1 &
sleep 15 
latest_dir=$(find ../data/ -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
mv log.out "$latest_dir/log.out"
