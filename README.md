# CRT-Net ECG Classification Project

This repository contains the implementation and experiments related to the preprocessing of ECG data from the MIT-BIH Arrhythmia Database and initial model training using CRT-Net architecture. This work is part of a larger project described in the [final group report](https://github.com/llevera/enhance_crt_net), which explores enhancements to the CRT-Net model for improved electrocardiogram (ECG) classification.

## Project Overview

ECG classification is a crucial task in medical diagnostics, as it involves interpreting the electrical activity of the heart to identify various types of arrhythmias. CRT-Net is a hybrid neural network architecture that combines Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformer models to capture both spatial and temporal features of ECG signals.

This repository focuses on the following aspects of the larger project:

- **Preprocessing MIT-BIH Arrhythmia Database**: Preparing the dataset for training, including handling missing data, normalizing signals, and segmenting the ECG records.
- **Initial Model Training**: Implementing and training the CRT-Net model on the preprocessed MIT-BIH dataset to establish a baseline performance.

## Usage

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/VladPlusIn/CRT-Net-ECG-Classification.git
   cd CRT-Net-ECG-Classification
   ```
## Data Preprocessing

The data preprocessing steps are critical for preparing the ECG signals for training. The notebook includes:

### Loading the MIT-BIH Arrhythmia Database

Importing ECG records and annotations.

```python
import wfdb
records = wfdb.get_record_list('mitdb')
```

### Segmenting ECG Signals

Extracting individual heartbeats from the ECG signals centered around the R-peak.

### Label Mapping
Mapping the annotations to the AAMI standard classes used for training.

```python
label_map = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
```
### Normalization

Normalizing the ECG signals to ensure consistent input for the model.

### Initial Model Training

The notebook includes the implementation of the CRT-Net model and its training on the preprocessed data:

1. Defining the CRT-Net model, which includes CNN, RNN, and Transformer components.
2. Setting up the training loop with the Adam optimizer and early stopping.
