#!/usr/bin/bash
declare -a arr1=("../out_split.pk" "out_split_wiki.pk")
declare -a arr3=("n","y")
declare -a arr2=("nbow","dan")

for i in "${arr1[@]}";
do
	for j in "${arr2[@]}";
	do
		for k in "${arr3[@]}";
		do
			python train.py -data $i -model $j -wd $k
		done
	done
done

