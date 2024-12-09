# Ad Sales Prediction - Logistic Regression

## Project Overview
This project implements a basic machine learning model using **Logistic Regression** to predict whether a customer will buy a product based on their **Age** and **Salary**. The dataset is preprocessed, scaled, and split for training and testing. The final model is evaluated for accuracy, and predictions are made for new customer data.

---

## Table of Contents
1. Dataset  
2. Steps Followed  
3. Technologies Used  
4. How to Run  
5. Results  
6. Future Improvements  

---

## Dataset
The dataset used in this project is `DigitalAd_dataset.csv`, containing the following columns:
- **Age**: Age of the customer  
- **EstimatedSalary**: Salary of the customer  
- **Purchased**: Target variable (1 if customer buys, 0 if they don't)  

---

## Steps Followed
1. **Data Loading and Exploration**
   - Loaded the dataset using `pandas`.
   - Explored the data using `.head()`, `.info()`, and `.describe()`.

2. **Splitting the Dataset**
   - Features (Age and Salary) were stored in **X**.
   - Target variable (**Purchased**) was stored in **Y**.
   - The dataset was split into **75% training** and **25% testing** using `train_test_split()`.

3. **Feature Scaling**
   - Scaled the feature values using **StandardScaler** for better model performance.

4. **Model Building and Training**
   - Built a **Logistic Regression** model using `sklearn.linear_model`.
   - Trained the model using the training data.

5. **Model Prediction and Evaluation**
   - Predicted outcomes on the test dataset.
   - Calculated model accuracy using **accuracy_score** from `sklearn.metrics`.

6. **Prediction for New Data**
   - The model predicts if a customer will buy or not based on **new Age and Salary inputs**.

---

## Technologies Used
- **Python**: Programming language  
- **Libraries**:  
   - **NumPy**: For numerical computations  
   - **Pandas**: For data manipulation  
   - **Scikit-learn**: For machine learning and model evaluation  

---

## How to Run
1. **Clone the Repository**  
git clone https://github.com/varssh16/mlproject_1.git
cd mlproject_1


2. **Install Dependencies**  
- Install the required libraries using pip:  
  ```
  pip install numpy pandas scikit-learn
  ```

3. **Run the Script**  
- Execute the script in a Python environment or Google Colab.  
- You can input **Age** and **Salary** when prompted for new customer predictions.

4. **Example Input/Output**  
- Input:  
  ```
  Enter Age of the new person: 32
  Enter Salary: 85000
  ```
- Output:  
  ```
  Customer will buy
  ```

---

## Results
The Logistic Regression model achieved:
- **Accuracy**: Approximately **80%**

### Sample Predictions
| Age | Salary  | Prediction        |
|-----|---------|-------------------|
| 32  | 85000   | Customer will buy |
| 25  | 40000   | Customer won't buy |

---

## Future Improvements
- Add more features to improve prediction accuracy.  
- Compare performance with other models like **Decision Trees** or **Random Forests**.  
- Perform hyperparameter tuning to optimize model performance.  

---

## Author
**VARSHINI VS**  
- GitHub: [Your GitHub Profile](https://github.com/varssh16)  
- LinkedIn: [Your LinkedIn Profile](www.linkedin.com/in/varsshini-vs-86b768288)
