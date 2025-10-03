Two models were evaluated in addition to EfficientNet. Their associated notebooks and pipelines are in this git repository.

Where can I find the loss metrics?
THey're all broken up like this?
'Loss/localization_loss': 0.05764798,
 'Loss/regularization_loss': 0.029542064,
 'Loss/total_loss': 0.5100068,
 'learning_rate': 0.00416}
 Are those classified as validation and training loss?

 
To improve maybe more data? I'm not sure it's so open ended, I need help minimizing that search space.

## SSD MobileNet Evaluation Metrics

| Metric Type       | IoU Threshold     | Area   | Max Detections | Value  |
|-------------------|------------------|--------|----------------|--------|
| Average Precision | 0.50:0.95        | all    | 100            | 0.061  |
| Average Precision | 0.50             | all    | 100            | 0.132  |
| Average Precision | 0.75             | all    | 100            | 0.053  |
| Average Precision | 0.50:0.95        | small  | 100            | 0.023  |
| Average Precision | 0.50:0.95        | medium | 100            | 0.213  |
| Average Precision | 0.50:0.95        | large  | 100            | 0.388  |
| Average Recall    | 0.50:0.95        | all    | 1              | 0.021  |
| Average Recall    | 0.50:0.95        | all    | 10             | 0.075  |
| Average Recall    | 0.50:0.95        | all    | 100            | 0.108  |
| Average Recall    | 0.50:0.95        | small  | 100            | 0.060  |
| Average Recall    | 0.50:0.95        | medium | 100            | 0.378  |
| Average Recall    | 0.50:0.95        | large  | 100            | 0.525  |


## EfficentDet



## Resnet Evaluation Metrics

| Metric Type       | IoU Threshold     | Area   | Max Detections | Value  |
|-------------------|------------------|--------|----------------|--------|
| Average Precision | 0.50:0.95        | all    | 100            | 0.041  |
| Average Precision | 0.50             | all    | 100            | 0.074  |
| Average Precision | 0.75             | all    | 100            | 0.040  |
| Average Precision | 0.50:0.95        | small  | 100            | 0.012  |
| Average Precision | 0.50:0.95        | medium | 100            | 0.142  |
| Average Precision | 0.50:0.95        | large  | 100            | 0.336  |
| Average Recall    | 0.50:0.95        | all    | 1              | 0.011  |
| Average Recall    | 0.50:0.95        | all    | 10             | 0.052  |
| Average Recall    | 0.50:0.95        | all    | 100            | 0.076  |
| Average Recall    | 0.50:0.95        | small  | 100            | 0.030  |
| Average Recall    | 0.50:0.95        | medium | 100            | 0.297  |
| Average Recall    | 0.50:0.95        | large  | 100            | 0.488  |

