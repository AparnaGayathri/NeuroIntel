# Neural Disease Predictor

## About

This webapp was developed using Flask Web Framework. The models used to predict the diseases were trained on large Datasets. All the links for datasets and the python notebooks used for model creation are mentioned below in this readme. The webapp can predict following Diseases:

- Alzheimer's disease
- Brain Stroke
- Brain Tumor

## Models with their Accuracy of Prediction
## Symptom Diagnosis
| Disease               | Type of Model            | Accuracy |
| --------------------- | ------------------------ | -------- |
| Alzheimer's disease   | Machine Learning Model   | 98.25%   |
| Brain Stroke          | Machine Learning Model   | 99.79%   |
| Brain Tumor           | Machine Learning Model   | 97%      |

## Brain Scan Analysis
| Disease               | Type of Model            | Accuracy |
| --------------------- | ------------------------ | -------- |
| Alzheimer's disease   | Deep Learning Model(CNN) | 79.80%   |
| Brain Stroke          | Deep Learning Model(CNN) | 99.38%   |
| Brain Tumor           | Deep Learning Model(CNN) | 91.58%   |

## Steps to run this application in your system

1. Clone or download the repo.
2. Open command prompt in the downloaded folder.
3. Create a virtual environment

```
mkvirtualenv environment_name
```

4. Install all the dependencies:

```
pip install -r requirements.txt
```

5. Run the application

```
python app.py
```
