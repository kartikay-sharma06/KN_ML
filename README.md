# Student Exam Performance Prediction App

A Machine Learning web application that predicts students' mathematics exam scores based on demographic, socioeconomic, and educational factors. The application integrates a trained machine learning model with a Flask-based web interface to provide predictions from user-input data.

## Overview

The Student Exam Performance Prediction App demonstrates how Machine Learning can be used to analyze student-related data and estimate academic performance. Users can enter relevant information through a web form and receive a predicted mathematics score.

The project covers the complete machine learning workflow, including data preprocessing, model training, prediction pipeline development, and web application integration.

## Features

- **Score Prediction:** Predicts a student's mathematics score using a trained ML model.
- **Interactive Web Interface:** Provides a form for entering student information.
- **Data Processing Pipeline:** Handles preprocessing and transformation of input features.
- **Model Integration:** Connects a trained machine learning model to a Flask application.
- **Student Information:** Supports student details such as name and class for reference.
- **Modular Architecture:** Separates data processing, model training, prediction, and application logic.
- **Prediction Results:** Displays the model's estimated score through the web interface.

## Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Model Serialization | Pickle / Joblib |
| Frontend | HTML, CSS |
| Development Tools | VS Code, Jupyter Notebook |

## Dataset Features

The application uses student performance data containing the following features:

- `gender`
- `race_ethnicity`
- `parental_level_of_education`
- `lunch`
- `test_preparation_course`
- `math_score`
- `reading_score`
- `writing_score`
- `total_score`
- `average_score`

The model uses the features selected during training to predict `math_score`. The target variable and input features must match the configuration used in the training pipeline.

## Project Structure

```text
Student-Exam-Performance-Prediction/
│
├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
│   ├── index.html
│   └── home.html
│
├── static/
│   └── style.css
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: This is a representative structure. Adjust the filenames and folders to match your actual repository.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/kartikay-sharma06/Student-Exam-Performance-Prediction.git
```

Navigate to the project directory:

```bash
cd Student-Exam-Performance-Prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If you use Command Prompt instead:

```cmd
venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure that `requirements.txt` includes all the libraries required by your application.

### 4. Run the Application

If `app.py` is your Flask entry point, execute:

```bash
python app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://127.0.0.1:5000/
```

## How to Use

1. Launch the Flask application.
2. Open the application in your browser.
3. Enter the required student information in the prediction form.
4. Submit the form to generate a prediction.
5. View the estimated mathematics score on the results page.

## Machine Learning Workflow

1. **Data Ingestion:** Load the student performance dataset.
2. **Data Preprocessing:** Handle categorical and numerical features using the configured preprocessing pipeline.
3. **Data Transformation:** Encode categorical variables and apply required numerical transformations.
4. **Model Training:** Train a regression model using the prepared dataset.
5. **Model Evaluation:** Evaluate the model using suitable regression metrics.
6. **Model Serialization:** Save the trained model and preprocessing object.
7. **Prediction Pipeline:** Transform incoming user data and generate predictions.
8. **Flask Integration:** Connect the prediction pipeline to the web interface.

## Potential Applications

- Estimating student academic performance.
- Exploring factors associated with exam scores.
- Demonstrating end-to-end Machine Learning deployment.
- Learning how to integrate Scikit-learn models into Flask applications.

## Limitations

- Predictions are estimates and are not guaranteed to match actual exam results.
- Performance depends on the quality and representativeness of the training dataset.
- The model identifies statistical patterns and does not establish that any particular feature causes a change in academic performance.
- The prediction should not be used as the sole basis for important educational decisions.

## Future Improvements

- Add model comparison and performance metrics to the dashboard.
- Display visualizations of predicted scores and relevant student statistics.
- Improve form validation and error handling.
- Add prediction history and downloadable reports.
- Deploy the application to a cloud hosting platform.
- Add model monitoring and periodic retraining.

## Author

**Kartikay Sharma**

BCA Student | Aspiring Software Engineer | Machine Learning Enthusiast

- GitHub: [kartikay-sharma06](https://github.com/kartikay-sharma06)
- LinkedIn: [Kartikay Sharma](https://www.linkedin.com/in/-sharmakartikay/)

## License

This project is intended for educational and learning purposes. Add a license file if you intend to distribute or reuse the project under specific terms.
