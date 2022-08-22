# Gait Feedback and Correction Generation Using Multivariate Signal Learning

## Description
- Pytorh GPU implementation of the Project `Gait Feedback Discovery and Correction Using Multivariate Time-Series Learning`
- Novel Neural Network Architecture Based on new Feedback Loss Function embedded


## Code
- All source code is in `source`.
- Train using the `main.py` file.
- Get model predictions using the `predict.py` file.

## Documentation
- Code is the documentation of itself.

## Usage
- Use `python3 main.py` to train your feedback discovery network.
- A summary of the pipeline is given in `report.pdf`.

## Demonstration
The pipeline is demonstrated below.

#- Training Curves.
#
#| Losses | Accuracies |
#| --- | --- |
#| ![](./figs/real_loss.PNG) | ![](./figs/real_accuracy.PNG) |
#
#- Classification Confusion Matrix.
#
#| Absolute Values | Percentages |
#| --- | --- |
#| ![](./figs/val_conf_mat.png) | ![](./figs/val_conf_mat_percent.png) |


- Feedback Discovery on a sample of 16-channel time series per 1042 time-step signal.

| Left Foot Vertical Acceleration [LAV]
| --- |
| ![](./results/pred.png)


## Contact
- apost035@umn.edu, trs.apostolou@gmail.com


