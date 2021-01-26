# pytorch-retinanet

Fork of yhenon/pytorch-retinanet. See their ReadMe for typical use.

Work for my internship at ENSTA Paristech.

Retinanet adapted for macro-classification based on micro detection. Here specifically for Architectural style classification based on detected architectural element.

Doesn't support SHAP yet.

At the moment it trains sequentially Detection and Classification and doesn't compute the GED at each iteration as the SHAP values are computed very slowly.

# Installation
for the organisation of the folder please follow the organisation propose in [here](https://github.com/JulesSanchez/architectural_style_classification)

# MONUAI
if you want to run the MONUAI train please write 'python train.py' with default parameters the code should work.
For inference and GED computation, please write  python inference.py --model_path 'yourcheckpoint.pt'

# Pascal Part
if you want to run the Pascal Part train please write 'python train_pascal.py' with default parameters the code should work.
 python inference.py --csv_inference 'data/test_Pascal_part.csv' --csv_train 'data/train_Pascal_part.csv' --csv_classes 'data/class_retinanet_Pascal.csv' --dataset 'PascalPart' --model_path 'yourcheckpoint.pt'


