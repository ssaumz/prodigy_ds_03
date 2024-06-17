# README: Task-03 - Data Science Internship at Prodigy InfoTech

## Project Overview

Excited to announce the successful completion of Task-03 during my data science internship at Prodigy InfoTech 🚀. This project involved crafting a decision tree classifier in Python using Jupyter Notebook to predict customer purchases based on demographic and behavioral data.

## Table of Contents

1. [Project Description](#project-description)
2. [Technologies Used](#technologies-used)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [License](#license)
7. [Contributors](#contributors)

## Project Description

Task-03 focused on building a decision tree classifier to predict whether customers would make a purchase based on their demographic and behavioral data. The project included data preprocessing, model training, evaluation, and visualization of the decision tree.

### Objectives

- Preprocess the dataset to handle missing values and encode categorical variables.
- Split the dataset into training and testing sets.
- Train a decision tree classifier on the training data.
- Evaluate the model's performance on the test data.
- Visualize the decision tree.

## Technologies Used

- **Python**: Programming language used for data manipulation, modeling, and visualization.
- **Jupyter Notebook**: Interactive computing environment for writing and running code.
- **Pandas**: Data manipulation library.
- **NumPy**: Numerical computing library.
- **Scikit-learn**: Machine learning library for model building and evaluation.
- **Matplotlib**: Visualization library for creating static plots.
- **Seaborn**: Statistical data visualization library based on Matplotlib.

## Setup and Installation

### Prerequisites

Ensure you have the following software installed:

- Python (version 3.6 or higher)
- Jupyter Notebook

### Running the Notebook

1. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Open the notebook file `Task-03-Decision-Tree-Classifier.ipynb` and run the cells sequentially.

## Usage

### Example Usage

The notebook provides a step-by-step guide on how to:

1. Load and preprocess the dataset.
2. Split the data into training and testing sets.
3. Train the decision tree classifier.
4. Evaluate the classifier's performance.
5. Visualize the decision tree.

#### Loading and Preprocessing Data

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv('data/customer_data.csv')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])

# Split the data into features and target
X = data.drop(columns=['purchase'])
y = data['purchase']
```

#### Splitting the Dataset

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### Training the Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
```

#### Evaluating the Classifier

```python
from sklearn.metrics import accuracy_score, classification_report

# Predict on the test data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
```

#### Visualizing the Decision Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.title('Decision Tree')
plt.show()
```

## Project Structure

```
prodigy-infotech-task-03/
├── data/
│   └── bank-full.csv  # Sample dataset
├── images/
│   └── decision_tree.png  # Example of generated decision tree plot
├── Task-03-prodigy_ds_03.ipynb  # Jupyter Notebook file
└── README.md  # Project documentation
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contributors

- **Saumya Poojari** - [saumya.poojarii7@gmail.com]
- LinkedIn - https://www.linkedin.com/in/ssaumz/

Feel free to reach out with any questions or feedback!

---

Thank you for your interest in this project. Happy coding! 🚀
