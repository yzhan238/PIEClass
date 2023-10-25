python src/PIEClass.py --data_dir ./datasets/AGNews/ \
                       --train dataset.txt \
                       --label_names classes.txt \
                       --test dataset.txt \
                       --test_labels labels.txt \
                       --prompt senti \
                       --freeze_layers 11 \
                       --gpu 0