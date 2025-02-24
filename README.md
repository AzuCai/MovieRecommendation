# Movie Recommendation System

This project implements a hybrid movie recommendation system using Collaborative Filtering (SVD) to predict user ratings and provide personalized recommendations. It is built using the MovieLens dataset and includes visual evaluation tools, such as rating distribution and prediction accuracy plots.

## Dataset

The dataset used in this project is the MovieLens 1M dataset, which contains 1 million ratings from users on various movies. You can download it from [MovieLens website](https://grouplens.org/datasets/movielens/1m/).

## Project Structure

```
Recommendation-System
├── data
│   └── ml-1m
│       ├── ratings.dat
│       └── movies.dat
├── src
│   ├── data_processing.py
│   ├── model.py
│   ├── main.py
│   ├── evaluation.py
│   └── visualization.py
└── README.md
```

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Installation

Install the required dependencies using the following command:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage

Run the example notebook using Jupyter:

```bash
jupyter notebook notebooks/example.ipynb
```

Alternatively, execute the scripts individually:

```bash
python src/data_processing.py
python src/model.py
python src/evaluation.py
python src/visualization.py
```

## Evaluation

The model is evaluated using Root Mean Square Error (RMSE), with lower values indicating better prediction accuracy.

## Visualizations

- Movie rating distribution
- Actual vs predicted ratings scatter plot

## Contributing

Feel free to contribute by opening issues or submitting pull requests for improvements.

## License

This project is licensed under the MIT License.
