# pytorch-retinanet

Fork of yhenon/pytorch-retinanet. See their ReadMe for typical use.

Work for my internship at ENSTA Paristech.

Retinanet adapted for macro-classification based on micro detection. Here specifically for Architectural style classification based on detected architectural element.

Doesn't support SHAP yet.

At the moment it trains sequentially Detection and Classification and doesn't compute the GED at each iteration as the SHAP values are computed very slowly.

runing 'train.py' with default parameters should work.
For inference and GED computation, you need at least a trained model.