CUDA_VISIBLE_DEVICES=0 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/A_FISCED_IRT_CDM_sensitive' --sensitive_name='fisced' --mode='sensitive'
CUDA_VISIBLE_DEVICES=0 python train_CDM.py --dataset_index=1 --save_path='/zjq/zhangdacao/pisa/save_new/B_FISCED_IRT_CDM_sensitive' --sensitive_name='fisced' --mode='sensitive'
CUDA_VISIBLE_DEVICES=0 python train_CDM.py --dataset_index=0 --save_path='/zjq/zhangdacao/pisa/save_new/A_FISCED_IRT_CDM_ours' --sensitive_name='fisced' --mode='ours'
CUDA_VISIBLE_DEVICES=0 python train_CDM.py --dataset_index=1 --save_path='/zjq/zhangdacao/pisa/save_new/B_FISCED_IRT_CDM_ours' --sensitive_name='fisced' --mode='ours'
