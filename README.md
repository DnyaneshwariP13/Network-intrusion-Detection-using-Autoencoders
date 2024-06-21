# Network-intrusion-Detection-using-Autoencoders


This project focuses on detecting anomalies in network traffic data using autoencoders. By leveraging deep learning techniques, the model aims to accurately identify unusual patterns that may indicate security threats or other issues.

## Table of Contents
- [Introduction](#introduction)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Visualizations](#visualizations)
- [Results](#results)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Introduction

Anomalies in network traffic can indicate security breaches or other significant issues. This project uses autoencoders to detect such anomalies by analyzing network traffic data. The dataset used is the `Train_data.csv`, which contains various features representing network activities.

## Data Cleaning and Preprocessing

Data cleaning and preprocessing steps include:

1. **Removing Unnecessary Columns**: Columns with zero variance and identical values are dropped.
2. **Handling Missing and Infinite Values**: Missing and infinite values are replaced or removed.
3. **Label Encoding**: The target variable `class` is encoded into binary values (1 for anomaly and 0 for normal).
4. **Feature Scaling**: Numeric features are scaled using `StandardScaler`.

```python
def data_cleaning(df):
    # Data cleaning steps
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    return df
```

## Model Architecture

The autoencoder model is built using TensorFlow and Keras. The architecture includes multiple dense layers for encoding and decoding the input data.

```python
input_layer = Input(shape=(INPUT_SHAPE,))
x = Dense(64, activation='relu')(input_layer)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(8, activation='relu')(x)
code = Dense(CODE_DIM, activation='relu')(x)
x = Dense(16, activation='relu')(code)
x = Dense(32, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output_layer = Dense(INPUT_SHAPE, activation='relu')(x)

autoencoder = Model(input_layer, output_layer, name='anomaly')
```

## Model Training

The model is trained using the Adam optimizer and Mean Absolute Error (MAE) loss function. Early stopping and model checkpointing are used to optimize training.

```python
autoencoder.compile(loss='mae', optimizer=Adam())
history = autoencoder.fit(X_train_genuine, X_train_genuine,
                          epochs=25, batch_size=256,
                          validation_data=(X_test, X_test),
                          callbacks=[checkpoint, earlystopping], shuffle=True)
```

## Evaluation

The model's performance is evaluated using various metrics, including accuracy, precision, recall, and the confusion matrix.

```python
reconstructions = autoencoder.predict(X_test, verbose=0)
reconstruction_error = mae(reconstructions, X_test)
classification_report(recons_df['y_true'], recons_df['y_pred'])
confusion_matrix(recons_df['y_true'], recons_df['y_pred'])
```

## Visualizations

Various visualizations are created to understand the model's performance and error distribution.

- **Training and Validation Loss**:
  ![Loss by Epoch](loss_by_epoch.png)

- **Error Distribution**:
  ![Error Distribution](error_distribution.png)

- **Confusion Matrix**:
  ![Confusion Matrix](confusion_matrix.png)

## Results

The autoencoder model demonstrates a strong capability to identify anomalies in network traffic with high recall and accuracy.

- **Recall Score**: `xx.x%`
- **Accuracy Score**: `xx.x%`

## Usage

To use the model for anomaly detection:

1. Clone the repository.
2. Ensure all dependencies are installed.
3. Run the data cleaning and preprocessing steps.
4. Train the autoencoder model.
5. Evaluate the model on test data.

```bash
git clone <repository_url>
cd <repository_folder>
pip install -r requirements.txt
python anomaly_detection.py
```

## Conclusion

This project successfully demonstrates the use of autoencoders for anomaly detection in network traffic data. The model effectively identifies unusual patterns, providing a valuable tool for enhancing network security.

---

Feel free to reach out for any questions or contributions.
