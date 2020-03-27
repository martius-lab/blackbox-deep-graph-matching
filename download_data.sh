#!/bin/bash


mkdir ./data/downloaded
mkdir ./data/downloaded/PascalVOC
mkdir ./data/downloaded/WILLOW
mkdir ./data/downloaded/SPair-71k

cd ./data/downloaded/PascalVOC

echo -e "\e[1mGetting PascalVOC annotations...\e[0m"
wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz
tar xzf voc2011_keypoints_Feb2012.tgz
echo -e "\e[32m... done\e[0m"

echo -e "\e[1mGetting PascalVOC data\e[0m"
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar
tar xf VOCtrainval_25-May-2011.tar
mv TrainVal/VOCdevkit/VOC2011 ./
rmdir TrainVal/VOCdevkit
rmdir TrainVal
echo -e "\e[32m... done\e[0m"

echo -e "\e[1mGetting WILLOW data\e[0m"
cd ../WILLOW
wget http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip
unzip WILLOW-ObjectClass_dataset.zip
echo -e "\e[32m... done\e[0m"


echo -e "\e[1mGetting SPair-71k data\e[0m"
cd ../SPair-71k
wget http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
tar xzf SPair-71k.tar.gz
mv SPair-71k/* ./
echo -e "\e[32m... done\e[0m"