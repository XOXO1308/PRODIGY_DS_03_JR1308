# Decision Tree Classifier: Bank Marketing Dataset

## Overview
This project demonstrates the implementation of a Decision Tree Classifier to predict whether a customer will subscribe to a term deposit using the bank marketing dataset from the UCI Machine Learning Repository. The project includes data preprocessing, model training, evaluation, and visualization, all implemented in Python. A detailed video explanation accompanies the code, walking you through each step and function.

## Features
- **Data Preprocessing:** Handling categorical variables and data splitting.
- **Model Training:** Building and training a Decision Tree Classifier.
- **Model Evaluation:** Evaluating the model using accuracy, classification report, confusion matrix, and cross-validation.
- **Feature Importance:** Identifying and displaying the importance of features.
- **Visualization:** Plotting and saving the Decision Tree structure.
- **Video Explanation:** A comprehensive video guide explaining the code and results.

## Project Structure
```
├── data/
│   └── bank-full.csv              # Dataset used for training and testing
├── src/
│   └── decision_tree_classifier.py  # Python script with the Decision Tree implementation
├── results/
│   └── decision_tree.png           # Saved visualization of the Decision Tree
├── videos/
│   └── decision_tree_explanation.mp4 # Video explanation of the code and results
├── README.md                       # Project description and instructions
└── requirements.txt                # Python dependencies
```

## Installation
To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/decision-tree-classifier.git
   cd decision-tree-classifier
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Open the Project in VS Code:**
   - Launch VS Code and open the project directory.

2. **Run the Script:**
   - Run the `decision_tree_classifier.py` script using the integrated terminal in VS Code:
     ```bash
     python src/decision_tree_classifier.py
     ```

3. **Follow the Video Explanation:**
   - Watch the `decision_tree_explanation.mp4` in the `videos/` directory for a detailed walkthrough of the code. The video explains each step of the script and its corresponding results, providing a clear understanding of the implementation.

4. **Explore the Code:**
   - The script includes detailed comments explaining each step, from importing libraries to evaluating the model.

## Key Sections of the Code
- **Importing Libraries:**
  - `pandas`, `numpy`, `scikit-learn`, and `matplotlib` for data manipulation, model building, and visualization.
  
- **Loading the Dataset:**
  - Data is loaded from `data/bank-full.csv`.

- **Data Exploration:**
  - Basic exploration using `.head()` and `.info()` to understand the dataset structure.

- **Feature Engineering:**
  - One-hot encoding is applied to categorical variables using `pd.get_dummies()`.

- **Model Training:**
  - A Decision Tree Classifier is built with specified hyperparameters and trained on the training data.

- **Model Evaluation:**
  - The model is evaluated using cross-validation, accuracy score, classification report, and confusion matrix.

- **Feature Importance:**
  - The importance of each feature is calculated and displayed.

- **Visualization:**
  - The Decision Tree is visualized and saved as a PNG file in the `results/` directory.

## Results
After running the script, you'll see:
- **Model Performance:** Cross-validation accuracy, detailed classification report, and confusion matrix.
- **Feature Importances:** Ranked list of the most important features.
- **Decision Tree Visualization:** A visual representation of the Decision Tree.

## Contribution
Feel free to fork this repository, make improvements, and submit a pull request. Contributions are welcome!

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgements
This project uses a dataset from the UCI Machine Learning Repository. Special thanks to Moro et al. (2011) for providing the dataset.

**Citation:**
Moro, S., Laureano, R., & Cortez, P. (2011). Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. In P. Novais et al. (Eds.), *Proceedings of the European Simulation and Modelling Conference - ESM'2011* (pp. 117-121). EUROSIS. Guimarães, Portugal. Available at: [pdf](http://hdl.handle.net/1822/14838) | [bib](http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt)
