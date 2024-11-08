
# Project Title

This project involves the analysis and prediction of patient profiles using machine learning models, specifically focusing on follicle data. The project utilizes Python and various libraries for data manipulation, visualization, and model training. **Please note that the data used in this project is simulated and not real patient data.**

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/belapyc/FolliclesPublic.git
    cd FolliclesPublic
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Load the necessary data files:
    - `cycles.pkl`: Contains the simulated dummy cycle data.
    - `best_params_all_scans.pkl`: Contains the best parameters for the Random Forest models.

2. Run the Jupyter notebook `predict_vis.ipynb` to execute the analysis and model training steps.

3. Key steps in the notebook:
    - Import necessary libraries and modules.
    - Load and preprocess the dummy data.
    - Train Random Forest models using the best parameters.
    - Train and predict using the `HistostepModel2` model.
    - Visualize the results using the `plot_patient` function.


