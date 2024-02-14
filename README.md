# Hotel Booking Cancellation Prediction App using Flask (Python)

## Introduction
This project involves building a machine learning model to predict hotel booking cancellations based on various features using the Hotel Booking Cancellation dataset. The goal is to create a Flask web application to deploy the model and allow users to input relevant data for prediction.

## Problem Statement
The primary objective is to develop a Flask API and deploy it on Heroku to predict whether a hotel booking will be canceled or not based on input features such as lead time, average price, special requests, and market segment type.

## Dataset
The data used in this project is sourced from the Hotel Booking Cancellation dataset, available on Kaggle. It contains information about hotel bookings, including features like lead time, average price, special requests, and more.

You can access the dataset on Kaggle via the following link:
[Hotel Booking Cancellation Dataset](https://www.kaggle.com/datasets/youssefaboelwafa/hotel-booking-cancellation-prediction)

You can access the Model via the following link:
[Hotels Booking Cancellation Flask APP](https://hotels-booking-prediction.onrender.com)

## Project Structure
- **app.py**: Flask application script containing API endpoints and model deployment logic.
- **model.py**: Data pre-processing and model selection.
- **templates/**: Directory containing HTML templates for web interface.
- **models/**: Directory containing the save models.
- **clf.pkl**: Serialized machine learning model (Gradient Boosting Classifier).
- **scaler.pkl**: Serialized scaler object for feature scaling.
- **requirements.txt**: File listing all project dependencies.
- **booking.csv**: The comma seperated value data which is processed here. 

## Project Dependencies
The following Python libraries are used in this project:
- Flask
- scikit-learn
- pandas
- numpy

## Access the Complete Project
The complete project can be accessed on GitHub via the following link:
[Hotel Booking Cancellation Prediction](https://github.com/example/hotel-booking-cancellation-prediction)

## References
- Hotel Booking Cancellation dataset: [Kaggle](https://www.kaggle.com/datasets/youssefaboelwafa/hotel-booking-cancellation-prediction)
- Flask documentation: [Flask](https://flask.palletsprojects.com/en/2.0.x/)
- scikit-learn documentation: [scikit-learn](https://scikit-learn.org/stable/)
- Pandas documentation: [Pandas](https://pandas.pydata.org/docs/)
- NumPy documentation: [NumPy](https://numpy.org/doc/)
