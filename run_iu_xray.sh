python main.py \
    --image_dir data/iu_xray/images/ \
    --ann_path data/iu_xray/annotation.json \
    --label_path data/iu_xray/labels.pickle \
    --img_init_protypes_path data/iu_xray/init_protypes_512_empty_224_both.pt \
    --text_init_protypes_path data/iu_xray/text_empty_initprotypes_512.pt \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --epochs 30 \
    --batch_size 12 \
    --lr_ve 1e-3 \
    --lr_ed 2e-3 \
    --step_size 10 \
    --gamma 0.8 \
    --num_layers 3 \
    --topk 8 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --seed 7580 \
    --beam_size 3 \
    --save_dir results/iu_xray/ \
    --log_period 50 \
    --n_gpu 1 \
    --weight_img_con_loss 4 \
    --num_cluster 40 \
    --img_num_protype 8 \
    --text_num_protype 4 \
    --con_margin 0.4 \
    --weight_bce_loss 1 \
    --gbl_num_cluster 8 \
    --weight_txt_con_loss 1 \

