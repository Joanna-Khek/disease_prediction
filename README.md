# Disease Prediction Web Application
Link to application: https://symptoms-prediction.herokuapp.com/


![diagnoseIt_logo](https://user-images.githubusercontent.com/53141849/182336864-60a42b8b-a830-492f-a302-8992532471ac.png)   

# Project Description
**DiagnoseIt** is a web application to help users predict the possible diseases from their symptoms. 
More often than not, we spend time googling what is wrong with us when we experience some symptoms. I wanted to create a consolidated site where users can select their symptoms from a list and immediately be well aware of the possible diseases that they might have. Upon identification of the possible diseases, users are provided with description of the disease as well as the precautions to prevent further damage.

# Project Demo
![diagnoseit_gif](https://user-images.githubusercontent.com/53141849/182337320-ea6b94f9-3637-4305-a089-bc8fecbb99d3.gif)

# Project Details
Data was obtained from https://www.kaggle.com/datasets/karthikudyawar/disease-symptom-prediction.
There are 306 observations, 135 symptoms and 42 target classes (diseases).
A random forest classifier was used to train the model and a F1 score of 0.968 was obtained. MLflow and hyperopt were used to tune the parameters.

The web application uses the flask framework and hosted on heroku


