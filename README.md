# ResNet Below 5 Million

This is the github repository for "Presumably Great Models Went Wrong: What Did We Learn in a Failed 
[Kaggle Competition](https://www.kaggle.com/competitions/deep-learning-spring-2025-project-1)". 

## Experiments Visualization

This project requires [wandb](https://github.com/wandb/wandb) to track experiment records 
given the high amount of cross-validations. 

```bash
pip install wandb
```

After installing wandb, login using

```bash
wandb login
```

## Predict on the Kaggle scoring set using the trained model

```bash
python predict.py
```
This will generate a "prediction.csv" in the current folder.

## Train

The training process has 3 parts.
For the first part of training (CV on model structures), run

```bash
python train_cv1.py
```

This should generate a "scores_cv1.csv" in the current folder. The scores should look like this table, where each column represents a fold.

|  |  |  |  |  |
|----------|----------|----------|----------|----------|
| 0.9000 | 0.8972 | 0.9009 | 0.8911 | 0.8985 |
| 0.9111 | 0.9106 | 0.9119 | 0.9112 | 0.9168 |
| 0.9143 | 0.9120 | 0.9157 | 0.9057 | 0.9143 |
| 0.9102 | 0.9042 | 0.9092 | 0.9025 | 0.9130 |
| 0.9078 | 0.9032 | 0.9021 | 0.9026 | 0.9046 |

For the second part of training (CV on activation order and dropouts), run

```bash
python train_cv2.py
```
This should generate a "scores_cv2.csv" in the current folder. The scores should look like this table.

|          |          |          |          |          |
|----------|----------|----------|----------|----------|
| 0.9164   | 0.9158   | 0.9202   | 0.9112   | 0.9189   |
| 0.9168   | 0.9180   | 0.9178   | 0.9146   | 0.9187   |
| 0.9144   | 0.9102   | 0.9118   | 0.9100   | 0.9201   |
| 0.9207   | 0.9236   | 0.9195   | 0.9200   | 0.9241   |

For the third part of training run
```bash
python retrain.py
```
This traines the model used for predict.py

For the training loss and validation curve, please access the run-time link provided by [wandb](https://github.com/wandb/wandb).