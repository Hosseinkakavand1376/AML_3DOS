# train models

# Discriminative methods
# train with DGCNN backbone and SR1 dataset
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SR1 --src SR1 --loss CE 

# train with DGCNN backbone and SR2 dataset
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SR2 --src SR2 --loss CE 

# train with PointNet++ backbone and SR1 dataset
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name PN2_CE_SR1 --src SR1 --loss CE 

# train with PointNet++ backbone and SR2 dataset
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name PN2_CE_SR2 --src SR2 --loss CE 


# Representation and Distance Based Models
# train with DGCNN backbone and SR1 dataset
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_ARPL_CS_SR1 --src SR1 --loss ARPL --cs 

# train with DGCNN backbone and SR2 dataset
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_ARPL_CS_SR2 --src SR2 --loss ARPL --cs 

#train with PointNet++ backbone and SR1 dataset
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name PN2_ARPL_CS_SR1 --src SR1 --loss ARPL --cs 

# #train with PointNet++ backbone and SR2 dataset
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name PN2_ARPL_CS_SR2 --src SR2 --loss ARPL --cs 
