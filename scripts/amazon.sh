python src/PIEClass.py --data_dir ./datasets/amazon/ \
                       --train train.txt \
                       --label_names label_names.txt \
                       --test test.txt \
                       --test_labels test_labels.txt \
                       --prompt senti \
                       --freeze_layers 11 \
                       --max_len 200 \
                       --gpu 0