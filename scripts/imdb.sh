python src/PIEClass.py --data_dir ./datasets/imdb/ \
                       --train train.txt \
                       --label_names label_names.txt \
                       --test test.txt \
                       --test_labels test_labels.txt \
                       --prompt senti \
                       --freeze_layers 11 \
                       --gpu 0