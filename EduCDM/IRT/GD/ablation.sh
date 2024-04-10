# CUDA_VISIBLE_DEVICES=1 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/A_ESCS_IRT_CDM_without_cls' --sensitive_name='escs' --mode='without_cls' &
# CUDA_VISIBLE_DEVICES=2 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/A_ESCS_IRT_CDM_without_reverse' --sensitive_name='escs' --mode='without_reverse' &
# CUDA_VISIBLE_DEVICES=3 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/A_ESCS_IRT_CDM_without_fair' --sensitive_name='escs' --mode='without_fair' &
# CUDA_VISIBLE_DEVICES=4 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/A_ESCS_IRT_CDM_only_ce' --sensitive_name='escs' --mode='only_ce' &
# CUDA_VISIBLE_DEVICES=5 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/A_ESCS_IRT_CDM_with_cls' --sensitive_name='escs' --mode='with_cls' &
# CUDA_VISIBLE_DEVICES=6 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/A_ESCS_IRT_CDM_with_reverse' --sensitive_name='escs' --mode='with_reverse' &
# CUDA_VISIBLE_DEVICES=7 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/A_ESCS_IRT_CDM_with_fair' --sensitive_name='escs' --mode='with_fair' &
# CUDA_VISIBLE_DEVICES=0 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/A_ESCS_IRT_CDM_only_add' --sensitive_name='escs' --mode='only_add' &

CUDA_VISIBLE_DEVICES=0 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_cls/A_ESCS_IRT_CDM_1' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.5 1 &
CUDA_VISIBLE_DEVICES=1 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_cls/A_ESCS_IRT_CDM_2' --sensitive_name='escs' --mode='ours' --w 1 0.2 0.5 1 &
CUDA_VISIBLE_DEVICES=2 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_cls/A_ESCS_IRT_CDM_3' --sensitive_name='escs' --mode='ours' --w 1 0.3 0.5 1 &
CUDA_VISIBLE_DEVICES=3 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_cls/A_ESCS_IRT_CDM_4' --sensitive_name='escs' --mode='ours' --w 1 0.4 0.5 1 &
CUDA_VISIBLE_DEVICES=4 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_cls/A_ESCS_IRT_CDM_5' --sensitive_name='escs' --mode='ours' --w 1 0.5 0.5 1 &
CUDA_VISIBLE_DEVICES=5 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_cls/A_ESCS_IRT_CDM_6' --sensitive_name='escs' --mode='ours' --w 1 0.6 0.5 1 &
CUDA_VISIBLE_DEVICES=6 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_cls/A_ESCS_IRT_CDM_7' --sensitive_name='escs' --mode='ours' --w 1 0.7 0.5 1 &
CUDA_VISIBLE_DEVICES=7 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_cls/A_ESCS_IRT_CDM_8' --sensitive_name='escs' --mode='ours' --w 1 0.8 0.5 1 &
CUDA_VISIBLE_DEVICES=0 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_cls/A_ESCS_IRT_CDM_9' --sensitive_name='escs' --mode='ours' --w 1 0.9 0.5 1 &

CUDA_VISIBLE_DEVICES=1 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_rev/A_ESCS_IRT_CDM_1' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.1 1 &
CUDA_VISIBLE_DEVICES=2 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_rev/A_ESCS_IRT_CDM_2' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.2 1 &
CUDA_VISIBLE_DEVICES=3 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_rev/A_ESCS_IRT_CDM_3' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.3 1 &
CUDA_VISIBLE_DEVICES=4 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_rev/A_ESCS_IRT_CDM_4' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.4 1 &
CUDA_VISIBLE_DEVICES=5 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_rev/A_ESCS_IRT_CDM_5' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.5 1 &
CUDA_VISIBLE_DEVICES=6 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_rev/A_ESCS_IRT_CDM_6' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.6 1 &
CUDA_VISIBLE_DEVICES=7 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_rev/A_ESCS_IRT_CDM_7' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.7 1 &
CUDA_VISIBLE_DEVICES=0 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_rev/A_ESCS_IRT_CDM_8' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.8 1 &
CUDA_VISIBLE_DEVICES=1 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_rev/A_ESCS_IRT_CDM_9' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.9 1 &

CUDA_VISIBLE_DEVICES=2 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_fair/A_ESCS_IRT_CDM_1' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.5 0.1 &
CUDA_VISIBLE_DEVICES=3 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_fair/A_ESCS_IRT_CDM_2' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.5 0.2 &
CUDA_VISIBLE_DEVICES=4 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_fair/A_ESCS_IRT_CDM_3' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.5 0.3 &
CUDA_VISIBLE_DEVICES=5 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_fair/A_ESCS_IRT_CDM_4' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.5 0.4 &
CUDA_VISIBLE_DEVICES=6 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_fair/A_ESCS_IRT_CDM_5' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.5 0.5 &
CUDA_VISIBLE_DEVICES=7 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_fair/A_ESCS_IRT_CDM_6' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.5 0.6 &
CUDA_VISIBLE_DEVICES=0 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_fair/A_ESCS_IRT_CDM_7' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.5 0.7 &
CUDA_VISIBLE_DEVICES=1 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_fair/A_ESCS_IRT_CDM_8' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.5 0.8 &
CUDA_VISIBLE_DEVICES=2 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_fair/A_ESCS_IRT_CDM_9' --sensitive_name='escs' --mode='ours' --w 1 0.1 0.5 0.9 &

CUDA_VISIBLE_DEVICES=3 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_ce/A_ESCS_IRT_CDM_1' --sensitive_name='escs' --mode='ours' --w 0.1 0.1 0.5 1 &
CUDA_VISIBLE_DEVICES=4 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_ce/A_ESCS_IRT_CDM_2' --sensitive_name='escs' --mode='ours' --w 0.2 0.1 0.5 1 &
CUDA_VISIBLE_DEVICES=5 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_ce/A_ESCS_IRT_CDM_3' --sensitive_name='escs' --mode='ours' --w 0.3 0.1 0.5 1 &
CUDA_VISIBLE_DEVICES=6 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_ce/A_ESCS_IRT_CDM_4' --sensitive_name='escs' --mode='ours' --w 0.4 0.1 0.5 1 &
CUDA_VISIBLE_DEVICES=7 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_ce/A_ESCS_IRT_CDM_5' --sensitive_name='escs' --mode='ours' --w 0.5 0.1 0.5 1 &
CUDA_VISIBLE_DEVICES=0 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_ce/A_ESCS_IRT_CDM_6' --sensitive_name='escs' --mode='ours' --w 0.6 0.1 0.5 1 &
CUDA_VISIBLE_DEVICES=1 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_ce/A_ESCS_IRT_CDM_7' --sensitive_name='escs' --mode='ours' --w 0.7 0.1 0.5 1 &
CUDA_VISIBLE_DEVICES=2 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_ce/A_ESCS_IRT_CDM_8' --sensitive_name='escs' --mode='ours' --w 0.8 0.1 0.5 1 &
CUDA_VISIBLE_DEVICES=3 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/w_ce/A_ESCS_IRT_CDM_9' --sensitive_name='escs' --mode='ours' --w 0.9 0.1 0.5 1 &

CUDA_VISIBLE_DEVICES=3 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/Ablation_study/IRT/A_ESCS_IRT_CDM_without_fair*' --sensitive_name='escs' --mode='without_fair*'
