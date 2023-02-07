# Multi_Cure
In this WebApp you can find 10 diseases under 1 app. That would be very useful for the user to check. It has some details about the disease such as their symptoms and some details about the disease.
This webapp was developed using Flask Web Framework. The models used to predict the diseases were trained on large Datasets. All the links for datasets and the python notebooks used for model creation are mentioned below in this readme. The webapp can predict following Diseases:

Alzheimer
Breast Cancer
Brain Tumor
Covid-19
Diabetes 
Heart Disease
Kidney Disease
Liver Disease
Malaria
Pneumonia



![ss](https://user-images.githubusercontent.com/67755812/217238738-f28f759c-934a-4da7-9729-8306f66952b3.png)




# Accuracy of Prediction for all Diseases:


| Disease        | Accuracy      | Dataset Link                                                                                             | Models          |
| -------------  |:-------------:| :-------------------------------------------------------------------------------------------------------: | --------------: |
|  Alzheimer     | 87.68%        | [Alzheimer Data](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)         | Deep Learning| 
| Breast Cancer  | 98.25%        | [Breast cancer Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)                 | Deep Learning   |
| Brain Tumor    | 83.82%        | [Brain Tumor Data](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)  | Machine Learning|
| Covid-19       | 81.52%        | [Covid-19 Data](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset)                   | Deep Learning     |
| Diabetes       | 96.50%         | [Diabetes Data](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)                    | Machnine Learning|
| Heart Disease  | 85.25%        | [Heart Disease Data](https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction)              | Machine Learning |
| Kidney Disease | 99.00%        | [Kidney DIsease Data](https://www.kaggle.com/datasets/mansoordaku/ckdisease)                             | Machine Learning |
| Liver Disease  | 77.00%        | [Liver Disease Data](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records)                 | Machine Learning |
| Malaria        | 93.70%        | [Malaria Data](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)               | Deep Learning|
| Pneumonia      | 92.23%        | [Pneumonia Data](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)                | Deep Learning|



# Steps to run this application in your system

#### a. Clone or download the repo.
#### b. Open command prompt in the downloaded folder.
#### c. Create a virtual environment
   `mkvirtualenv environment_name`
#### d. Install all the dependencies:
`pip install -r requirements.txt`
#### e. Run the application
`python app.py`
