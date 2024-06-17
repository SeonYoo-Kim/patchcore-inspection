datapath="../dataset/AeBAD"
datasets=('background'  'illumination'  'same'  'view')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

# Ensemble: Backbone: WR101 / ResNext101/ DenseNet201, Blocks: 2 & 3, Coreset Percentage: 1%, Embedding Dimensionalities: 1024 > 384, neighbourhood aggr. size: 5, neighbours: 5, seed: 88
# Performance: Instance AUROC: 0.996, Pixelwise AUROC: 0.982, PRO: 0.949
python bin/run_patchcore.py --gpu 0 --seed 88 --save_patchcore_model --log_group AeBAD_IM480_WR101_L2-3_P001_D1024-1024_PS-5_AN-3_S88 --log_project AeBAD_Results results \
patch_core -b wideresnet101 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 --faiss_on_gpu --anomaly_scorer_num_nn 3 \
--patchsize 5 sampler -p 0.01 approx_greedy_coreset dataset --resize 512 --imagesize 480 "${dataset_flags[@]}" aebad $datapath