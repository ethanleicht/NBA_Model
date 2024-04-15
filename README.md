# Predicting NBA Outcomes with Machine Learning
Rashed Almogezwi, Ethan Leicht


In this repository, you will find work from outside sources:
- All files and folders in `parent_paper` are from the parent paper's Git repository, which can be accessed [here](https://github.com/mhoude1/NBA_Model).
- `nba_data/stats_archive` and `nba_data/games_archive` are downloaded from Kaggle, which can be accessed [here](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats?select=Advanced.csv).

The remainder of the files and folders are our work:
- `parent_paper_remodulation.py` is our remodulation of the parent paper's original code in `parent_paper/full_code.ipynb`.
- `make_df.ipynb` pulls data from `nba_data/stats_archive` and `nba_data/games_archive` to create `nba_data/nba_df_final.csv`.
- `nba_data/nba_df_final.csv` is the dataset used for feature/dimensionality reduction and machine learning in `reduce_train_analyze.ipynb`.
- `reduce_train_analyze.ipynb` generates visualizations of our results and saves the images in `plots`.
