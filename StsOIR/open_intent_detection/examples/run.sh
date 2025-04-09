#!/usr/bin bash

# param 2 weChat
mName="M199_09/testing"
mTitle="OIDDD_Finished!!"
mDataset="stackoverflow"
mKlr="0.75"
mLr="1.0"
mSeed="0-9"

for dataset in 'stackoverflow' 
do
    for known_cls_ratio in 0.25
    # for known_cls_ratio in 0.25 0.5 0.75 
    do
        for labeled_ratio in 0.25 0.5 1.0
        do
            for seed in 0 1 2 3 4 5 6 7 8 9
            do 
                python run.py \
                --dataset $dataset \
                --method 'OIDDD' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --backbone 'bert' \
                --config_file_name 'OIDDD' \
                --gpu_id '0' \
                --train \
                --save_results \
                --results_file_name 'results_OIDDD_bert_cluster.csv' 
            done
        done
    done
done

# postWeChatMsg.py
python postWeChatMsg.py --name $mName --title $mTitle --content "$mDataset $mKlr $mLr $mSeed"