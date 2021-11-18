python main.py \
    --image_dir data/iu_xray/images/ \
    --ann_path data/iu_xray/annotation.json \
    --label_path data/iu_xray/labels.pickle \
    --init_protypes_path data/iu_xray/init_protypes_512_duplicate_224_both.pt \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --epochs 50 \
    --batch_size 12 \
    --lr_ve 5e-4 \
    --lr_ed 1e-3 \
    --step_size 10 \
    --gamma 0.8 \
    --num_layers 3 \
    --topk 4 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --seed 7580 \
    --beam_size 3 \
    --save_dir results/iu_xray/ \
    --log_period 50 \
    --n_gpu 1 \
    --weight_cnn_loss 0.0 \
    --num_cluster 40 \
    --num_prototype 8 \

