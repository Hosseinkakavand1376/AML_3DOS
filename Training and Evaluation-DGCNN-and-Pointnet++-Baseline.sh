# train models

# Discriminative methods
# train with DGCNN backbone and SR1 dataset
python classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SR1 --src SR1 --loss CE 

# train with DGCNN backbone and SR2 dataset
python classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SR2 --src SR2 --loss CE 

# train with PointNet++ backbone and SR1 dataset
python classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name PN2_CE_SR1 --src SR1 --loss CE 

# train with PointNet++ backbone and SR2 dataset
python classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name PN2_CE_SR2 --src SR2 --loss CE 


# Representation and Distance Based Models
# train with DGCNN backbone and SR1 dataset
python classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_ARPL_CS_SR1 --src SR1 --loss ARPL --cs 

# train with DGCNN backbone and SR2 dataset
python classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_ARPL_CS_SR2 --src SR2 --loss ARPL --cs 

# train with DGCNN by OpenShape on SR1 dataset
python classifiers/trainer_openshape.py --config cfgs/dgcnn-cla.yaml --exp_name Openshabe B32-SR1 --src SR1 --loss ARPL --ckpt_path /content/model_last.pt > Openshabe B32-SR1.txt

# train with DGCNN by OpenShape on SR2 dataset
python classifiers/trainer_openshape.py --config cfgs/dgcnn-cla.yaml --exp_name Openshabe B32-SR2 --src SR2 --loss ARPL --ckpt_path /content/model_last.pt > Openshabe B32-SR2.txt

#train with PointNet++ backbone and SR1 dataset
python classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name PN2_ARPL_CS_SR1 --src SR1 --loss ARPL --cs 

# #train with PointNet++ backbone and SR2 dataset
python classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name PN2_ARPL_CS_SR2 --src SR2 --loss ARPL --cs 

## Evaluation

# # eval DGCNN backbone and SR1 dataset
python classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SR1_eval --src SR1 --loss CE -mode eval --ckpt_path outputs/DGCNN_CE_SR1/models/model_last.pth > DGCNN_CE_SR1_eval.txt

# # # eval DGCNN backbone and SR2 dataset
python classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SR2_eval --src SR2 --loss CE -mode eval --ckpt_path outputs/DGCNN_CE_SR2/models/model_last.pth > DGCNN_CE_SR2_eval.txt

# Eval with DGCNN by OpenShape on SR1 dataset
python classifiers/trainer_openshape.py --config cfgs/dgcnn-cla.yaml --exp_name Openshabe B32-SR1 --src SR1 --loss CE --ckpt_path /content/model_last.pt > Openshabe B32-SR1.txt

# Eval with DGCNN by OpenShape on SR2 dataset
python classifiers/trainer_openshape.py --config cfgs/dgcnn-cla.yaml --exp_name Openshabe B32-SR2 --src SR2 --loss CE --ckpt_path /content/model_last.pt > Openshabe B32-SR2.txt

# Failure case analysis on SR1 dataset
python classifiers/trainer_Failcases.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SR1_Failcases --src SR1 --loss CE -mode eval --ckpt_path /content/model_last.pth > Failcases_SR1_output.txt

# Failure case analysis on SR2 dataset
!python classifiers/trainer_Failcases.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SR2_Failcases --src SR2 --loss CE -mode eval --ckpt_path /content/model_last.pth > Failcases_SR2_output.txt