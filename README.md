# Boston Housing Price Prediction with Decision Tree Regression 🏡💰

## Project Overview

This repository presents a machine learning project focused on predicting Boston house prices using a Decision Tree Regressor. The goal was to build a robust and accurate model by leveraging hyperparameter tuning and cross-validation techniques to enhance its performance and generalization capabilities. 🚀

The project utilizes the [Boston Housing Price Dataset](https://www.kaggle.com/datasets/arunjathari/bostonhousepricedata) from Kaggle, a classic dataset in machine learning for regression tasks. 📊

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features Used](#features-used)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Selection](#model-selection)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Cross-Validation](#cross-validation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Dataset 🏘️

The dataset used in this project is the **Boston Housing Price Dataset**, available on Kaggle. It contains various socio-economic factors and housing attributes for census tracts in the Boston area. 📍

**Key features in the dataset include:**

* **CRIM:** Per capita crime rate by town 🚔
* **ZN:** Proportion of residential land zoned for lots over 25,000 sq.ft. 🌳
* **INDUS:** Proportion of non-retail business acres per town 🏭
* **CHAS:** Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 🌊
* **NOX:** Nitric oxides concentration (parts per 10 million) 💨
* **RM:** Average number of rooms per dwelling 🛏️
* **AGE:** Proportion of owner-occupied units built prior to 1940 ⏳
* **DIS:** Weighted distances to five Boston employment centers 🚶
* **RAD:** Index of accessibility to radial highways 🛣️
* **TAX:** Full-value property-tax rate per $10,000 💲
* **PTRATIO:** Pupil-teacher ratio by town 🧑‍🏫
* **B:** 1000(Bk - 0.63)^2 where Bk is the proportion of Blacks by town 👥
* **LSTAT:** % lower status of the population 📉
* **MEDV:** Median value of owner-occupied homes in $1000s (Target variable) 🎯

## Features Used 💡

All available features (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT) were used as independent variables to predict the `MEDV` (Median Value) target variable. ✨

## Methodology 🔬

### Data Preprocessing 🧹

Before training the model, the data was preprocessed to handle any missing values (if present) and scale the features. This step ensures that all features contribute equally to the model training process and prevents features with larger numerical ranges from dominating. ⚖️

### Model Selection 🌳

A **Decision Tree Regressor** was chosen for this project due to its interpretability and ability to capture non-linear relationships in the data. 🌲

### Hyperparameter Tuning ⚙️

To optimize the Decision Tree Regressor's performance and prevent overfitting, **hyperparameter tuning** was performed. Techniques such as `GridSearchCV` were employed to systematically search for the best combination of hyperparameters, including: 🧪

* `max_depth`: The maximum depth of the tree.
* `min_samples_split`: The minimum number of samples required to split an internal node.
* `min_samples_leaf`: The minimum number of samples required to be at a leaf node.
* `splitter`: The strategy used to choose the split at each node.

### Cross-Validation ✅

**K-Fold Cross-Validation** was integrated with the hyperparameter tuning process. This technique helps to assess the model's generalization performance by splitting the dataset into multiple folds, training the model on a subset of these folds, and validating on the remaining fold. This process is repeated for each fold, providing a more robust estimate of the model's performance than a single train-test split. 📈

## Results 🏆

After hyperparameter tuning and cross-validation, the model achieved optimal performance metrics (e.g., R-squared, Mean Squared Error) on the unseen data. The specific metrics and the best hyperparameters found are detailed within the project notebooks/scripts. 💯

**(Optional: You can add specific metrics here once you have them, e.g.)**
* **Best R-squared Score:** `[Your R-squared Value]` ⭐
* **Optimal Hyperparameters:**
    * `max_depth`: `[Value]`
    * `min_samples_split`: `[Value]`
    * `min_samples_leaf`: `[Value]`

## Installation 💻

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/pranavgaikwad988/Boston-Housing-Price-Prediction-with-Decision-Tree-Regression.git](https://github.com/pranavgaikwad988/Boston-Housing-Price-Prediction-with-Decision-Tree-Regression.git)
    cd Boston-Housing-Price-Prediction-with-Decision-Tree-Regression
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: `venv\Scripts\activate`
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    (You'll need to create a `requirements.txt` file in your project by running `pip freeze > requirements.txt` after installing all your libraries)

## Usage ▶️

1.  **Download the dataset:**
    Download `boston_house_prices.csv` from [Kaggle](https://www.kaggle.com/datasets/arunjathari/bostonhousepricedata) and place it in the `data/` directory (you might need to create this directory). 📂

2.  **Run the Jupyter Notebook/Python script:**
    Open and run the `[your_notebook_name].ipynb` (e.g., `boston_housing_decision_tree.ipynb`) notebook or execute the main Python script `[your_script_name].py` to see the full analysis, model training, and results. 🚀

    ```bash
    jupyter notebook
    # or
    python your_script_name.py
    ```

## Contributing 🤝

Contributions are welcome! If you have suggestions for improving the model, code, or documentation, please feel free to:

1.  Fork the repository. 🍴
2.  Create a new branch (`git checkout -b feature/AmazingFeature`). 🌿
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`). 📝
4.  Push to the branch (`git push origin feature/AmazingFeature`). 📤
5.  Open a Pull Request. ➡️

## License 📄

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact ✉️

Pranav Gaikwad - [gaikwadpranav988@gmail.com](mailto:gaikwadpranav988@gmail.com)

LinkedIn: [https://www.linkedin.com/in/pranav-gaikwad-0b94032a](https://www.linkedin.com/in/pranav-gaikwad-0b94032a)

Project Link: [https://github.com/pranavgaikwad988/Boston-Housing-Price-Prediction-with-Decision-Tree-Regression](https://github.com/pranavgaikwad988/Boston-Housing-Price-Prediction-with-Decision-Tree-Regression)
