python src/PIEClass.py --data_dir ./datasets/20News/ \
                       --train dataset.txt \
                       --label_names classes.txt \
                       --test dataset.txt \
                       --test_labels labels.txt \
                       --prompt topic \
                       --num_iter 8 \
                       --gpu 0