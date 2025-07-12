# Comparative Analysis between SVD and KNN in Recommender Systems

This project presents a comparative analysis between two recommendation system algorithms:

* **SVD (Singular Value Decomposition)** — matrix factorization (latent factors)
* **KNN (K-Nearest Neighbors)** — neighborhood-based collaborative filtering (item-based)

Using the **MovieLens 100k** dataset, the project evaluates both algorithms using RMSE and MAE metrics, applying 5-fold cross-validation and train/test split.

---

## Main File

* `main01.py`: Python script to load data, train, test, and evaluate the SVD and KNN models, and generate comparison plots.

---

## Requirements

* Python 3.6 or later
* Python libraries:

  * pandas
  * matplotlib
  * scikit-surprise (surprise)

---

## Installation

Run the command below to install the required dependencies:

```bash
pip install pandas matplotlib scikit-surprise
```

---

## How to Use

1. Clone this repository or download the `main01.py` file.
2. Run the script:

```bash
python main01.py
```

The script will:

* Load the MovieLens 100k dataset
* Split the data into training and testing sets
* Train and evaluate the SVD and KNN (item-based) models
* Perform 5-fold cross-validation for comparison
* Display RMSE and MAE metrics in the console
* Generate comparison plots of performance across folds

---

## Expected Results

* RMSE and MAE metrics for both algorithms on the test set
* Comparative graphs showing the performance of SVD and KNN over the folds

---

## References

* MovieLens 100k dataset: [https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/)
* Surprise Library: [http://surpriselib.com/](http://surpriselib.com/)
* Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30-37.

---

## Contact

For questions or contributions, open an issue or submit a pull request.
