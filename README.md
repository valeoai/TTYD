<div align='center'>

# Train Till You Drop: Towards Stable and Robust Source-free Unsupervised 3D Domain Adaptation

[Björn Michele](https://bjoernmichele.com)<sup>1,3</sup>&nbsp;&nbsp;
[Alexandre Boulch](https://boulch.eu/)<sup>1</sup>&nbsp;&nbsp;&nbsp;
[Tuan-Hung Vu](https://tuanhungvu.github.io/)<sup>1</sup>&nbsp;&nbsp;&nbsp;
[Gilles Puy](https://sites.google.com/site/puygilles/)<sup>1</sup>&nbsp;&nbsp;&nbsp;
[Renaud Marlet](http://imagine.enpc.fr/~marletr/)<sup>1,2</sup>&nbsp;&nbsp;
[Nicolas Courty](https://people.irisa.fr/Nicolas.Courty/)<sup>3</sup>&nbsp;&nbsp;&nbsp;

<sub>
<sup>1</sup> Valeo.ai, Paris, France&nbsp;
<sup>2</sup> LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS, Marne-la-Vallée, France

<sup>3</sup> CNRS, IRISA, Univ. Bretagne Sud, Vannes, France
</sub>

<br/>

[![Arxiv](https://img.shields.io/badge/paper-arxiv.2409.04409-B31B1B.svg)](https://arxiv.org/abs/2409.04409)

Accepted at ECCV 2024
</div>
<br/>


## Abstract
We tackle the challenging problem of source-free unsupervised domain adaptation (SFUDA) for 3D semantic segmentation. It amounts to performing domain adaptation on an unlabeled target domain without any access to source data; the available information is a model trained to achieve good performance on the source domain. A common issue with existing SFUDA approaches is that performance degrades after some training time, which is a by-product of an under-constrained and ill-posed problem. We discuss two strategies to alleviate this issue. First, we propose a sensible way to regularize the learning problem. Second, we introduce a novel criterion based on agreement with a reference model. It is used (1) to stop the training when appropriate and (2) as validator to select hyperparameters without any knowledge on the target domain. Our contributions are easy to implement and readily amenable for all SFUDA methods, ensuring stable improvements over all baselines. We validate our findings on various 3D lidar settings, achieving state-of-the-art performance.

---

## Dependencies

This code was implemented and tested with python 3.10, PyTorch 1.13.1 and CUDA 11.7.
The MinkUnet backbone is implemented with version 1.4.0 of [Torchsparse](https://github.com/mit-han-lab/torchsparse.)([Exact commit](https://github.com/mit-han-lab/torchsparse/commit/69c1034ddb285798619380537802ea0ff03aeba6)).



---

# Training 

## Datasets 
The datasets should be placed in data/

## Source-models
Please find the source-models we start from in the [model zoo](#Source-models). 

We explain it here exemplaric for nuScenes to SemanticKITTI. For other combinations, please change the --setting command (NS2SK, Synth2SK, Synth2POSS, NS2POSS, NS2PD, NS2WY). In case the source dataset is SyntheticLiDAR the path with the --resume_path parameter also has to be adapted. 

## TTYD-Core

``python train_ttyd_core.py --name="TTYD_core_ns_sk"  --bn_layer="scaling_per_channel" --resume_path=source_models/ns_semantic_TorchSparseMinkUNet --setting='NS2SK' --learning_rate=0.00001  --ent_loss_thr=0.02  --div_loss_thr=0.02  --tensorboard_folder='TTYD_core'``

---

## Validator & Evaluation

``python class_agree_evaluator.py --name="TTYD_stop_agree_ns_sk" --resume_path='model_zoo/TTYD_Core/TTYD_Core_before_selection_ns_sk'``


---

## Self-Training 

``python train_ttyd_st.py --name="TTYD_self_training_ns_sk"  --bn_layer="scaling_per_channel" --resume_path='model_zoo/TTYD_Core/TTYD_Core_before_selection_ns_sk/model_4000.pth' --setting='NS2SK' --finetune=True --tensorboard_folder='TTYD_ST' --fintune_setting='complete_finetune' --pl_no_mapping=True --fintune_setting='classic' --lr_scheduler=True --learning_rate=0.0025`` 


# Model Zoo

## Source models
We start from the following 2 source models:

| Source datatset | Link | 
| -----           | ------- |
|     NuScenes    | [Link](https://drive.google.com/drive/folders/1NpjvWzo7agNtLFu6HODRhIElTP3a04n7?usp=drive_link)|
| SyntheticLiDAR  | [Link](https://drive.google.com/drive/folders/1NrFpTUYmlmBBHqjyAolvp9FAoBSdvdoa?usp=drive_link) |

## TTYD Core models 

| Setting | Link | 
| -----   | ------- |
| NS2SK   | TBD|
| SL2SK  | TBD |
| SL2SP  | TBD |
| NS2SP  | TBD |
| NS2PD  | TBD |
| NS2WO  | TBD |


## Acknowledgments

For the Self-Training code in the file "learn_mapping_ur_data.py" we rely on the code of [DT-ST](https://github.com/DZhaoXd/DT-ST). We thank them for making their work publicily available.

We also acknowledge the support of the French Agence Nationale de la Recherche (ANR), under grants ANR-21-CE23-0032 (project MultiTrans), ANR-20-CHIA-0030(OTTOPIA AI chair), and the European Lighthouse on Secure and Safe AI funded by the European Union under grant agreement No. 101070617. This work was performed using HPC resources from GENCI–IDRIS (2022-AD011013839,2023-AD011013839R1).

---

