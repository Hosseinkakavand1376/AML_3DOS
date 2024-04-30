# Evaluation models

# Discriminative methods

# Make directory for evaluations
mkdir -p Evaluations

# # eval DGCNN backbone and SR1 dataset
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SR1_eval --src SR1 --loss CE -mode eval --ckpt_path outputs/DGCNN_CE_SR1/models/model_last.pth >> Evaluations/DGCNN_CE_SR1_eval.txt

# # # eval DGCNN backbone and SR2 dataset
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SR2_eval --src SR2 --loss CE -mode eval --ckpt_path outputs/DGCNN_CE_SR2/models/model_last.pth >> Evaluations/DGCNN_CE_SR2_eval.txt

# # eval PointNet++ backbone and SR1 dataset
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name PN2_CE_SR1_eval --src SR1 --loss CE -mode eval --ckpt_path outputs/PN2_CE_SR1/models/model_last.pth >> Evaluations/PN2_CE_SR1_eval.txt

# # eval with PointNet++ backbone and SR2 dataset
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name PN2_CE_SR2_eval --src SR2 --loss CE -mode eval --ckpt_path outputs/PN2_CE_SR2/models/model_last.pth >> Evaluations/PN2_CE_SR2_eval.txt


# # Representation and Distance Based Models

# ##ARPL+CS
# # eval DGCNN backbone and SR1 dataset
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SR1_eval --src SR1 --loss ARPL --cs -mode eval --ckpt_path outputs/DGCNN_ARPL_CS_SR1/models/model_last.pth >> Evaluations/DGCNN_CE_SR1_eval.txt

# # eval DGCNN backbone and SR2 dataset
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SR2_eval --src SR2 --loss ARPL --cs -mode eval --ckpt_path outputs/DGCNN_ARPL_CS_SR2/models/model_last.pth >> Evaluations/DGCNN_CE_SR2_eval.txt

# # eval PointNet++ backbone and SR1 dataset
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name PN2_CE_SR1_eval --src SR1 --loss ARPL -mode eval --ckpt_path outputs/PN2_CE_SR1/models/model_last.pth >> Evaluations/PN2_CE_SR1_eval.txt

# # eval with PointNet++ backbone and SR2 dataset
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name PN2_CE_SR2_eval --src SR2 --loss ARPL -mode eval --ckpt_path outputs/PN2_CE_SR2/models/model_last.pth >> Evaluations/PN2_CE_SR2_eval.txt
