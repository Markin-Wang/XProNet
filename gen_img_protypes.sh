torchrun --nnodes=1 --nproc_per_node=1 gen_img_protypes.py \
    --image_dir data/mimic_cxr/images/ \
    --ann_path data/mimic_cxr/annotation.json \
    --label_path data/mimic_cxr/labels_14.pickle \
    --init_protypes_path data/cxr_gnome/init_prototypes.pt \
    --dataset_name mimic_cxr \
    --max_seq_length 100 \
    --threshold 10 \
    --epochs 30 \
    --batch_size 32 \
    --lr_ve 5e-5 \
    --lr_ed 1e-4 \
    --step_size 3 \
    --num_layers 3 \
    --topk 15 \
    --seed 7580  \
    --save_dir results/cxr_gnome/ \
    --log_period 500 \
    --n_gpu 1 \
    --num_cluster 14 \
    --img_con_margin 0.4 \
    --txt_con_margin 0.4 \
    --weight_txt_con_loss 0.1 \
    --weight_img_con_loss 1 \
    --d_img_ebd 2048 \
    --d_txt_ebd 768 \
    --num_protype 20 \
    --use_amp \
    --num_workers 12 \
