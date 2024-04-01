import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import pickle



def get_raw_data():
    df_2018 = pd.read_csv('nba_data/nba_df_2018.csv')
    df_2018['Date'] = pd.to_datetime(df_2018['Date'])
    df_2018['Season'] = '2018-19'

    df_2019 = pd.read_csv('nba_data/nba_df_2019.csv')
    df_2019['Date'] = pd.to_datetime(df_2019['Date'])
    df_2019['Season'] = '2019-20'

    df_2019_2 = pd.read_csv('nba_data/nba_df_2019_2.csv')
    df_2019_2['Date'] = pd.to_datetime(df_2019_2['Date'])
    df_2019_2['Season'] = '2019-20'

    frames = [df_2019, df_2019_2]
    df_2019_final = pd.concat(frames)
    len(df_2019_final)

    df_2020 = pd.read_csv('nba_data/nba_df_2020.csv')
    df_2020['Date'] = pd.to_datetime(df_2020['Date'])
    df_2020['Season'] = '2020-21'
    len(df_2020)

    frames = [df_2018, df_2019_final, df_2020]
    df = pd.concat(frames)
    df = df.reset_index(drop=True)

    return df



def get_avg_win_pct_last_n_games(team, game_date, df, n):
        prev_game_df = df[df['Date'] < game_date][(df['Home'] == team) | (df['Away'] == team)].sort_values(by = 'Date').tail(n)
        
        wins = 0 
        
        result_df = prev_game_df.iloc[:, range(0,32,31)]
        h_df = result_df.loc[result_df['Home'] == team] 
        
        h_wins = h_df.loc[h_df['Result'] == 1]
        
        wins += len(h_wins)
        
        a_df = result_df.loc[result_df['Home'] != team]
        a_wins = a_df.loc[a_df['Result'] == 0]
        
        wins += len(a_wins)

        return wins/n



# Home and road team win probabilities implied by Elo ratings and home court adjustment 
import math
def win_probs(home_elo, away_elo, home_court_advantage) :
    h = math.pow(10, home_elo/400)
    r = math.pow(10, away_elo/400)
    a = math.pow(10, home_court_advantage/400) 

    denom = r + a*h
    home_prob = a*h / denom
    away_prob = r / denom 
  
    return home_prob, away_prob

#odds the home team will win based on elo ratings and home court advantage
def home_odds_on(home_elo, away_elo, home_court_advantage) :
    h = math.pow(10, home_elo/400)
    r = math.pow(10, away_elo/400)
    a = math.pow(10, home_court_advantage/400)
    return a*h/r

#this function determines the constant used in the elo rating, based on margin of victory and difference in elo ratings
def elo_k(MOV, elo_diff):
    k = 20
    if MOV>0:
        multiplier=(MOV+3)**(0.8)/(7.5+0.006*(elo_diff))
    else:
        multiplier=(-MOV+3)**(0.8)/(7.5+0.006*(-elo_diff))
    return k*multiplier

#updates the home and away teams elo ratings after a game 
def update_elo(home_score, away_score, home_elo, away_elo, home_court_advantage) :
    home_prob, away_prob = win_probs(home_elo, away_elo, home_court_advantage) 

    if (home_score - away_score > 0) :
        home_win = 1 
        away_win = 0 
    else :
        home_win = 0 
        away_win = 1 
  
    k = elo_k(home_score - away_score, home_elo - away_elo)

    updated_home_elo = home_elo + k * (home_win - home_prob) 
    updated_away_elo = away_elo + k * (away_win - away_prob)
    
    return updated_home_elo, updated_away_elo

#takes into account prev season elo
def get_prev_elo(team, date, season, team_stats, elo_df) :
    prev_game = team_stats[team_stats['Date'] < date][(team_stats['Home'] == team) | (team_stats['Away'] == team)].sort_values(by = 'Date').tail(1).iloc[0] 

    if team == prev_game['Home'] :
        elo_rating = elo_df[elo_df['Game_ID'] == prev_game['Game_ID']]['H_Team_Elo_After'].values[0]
    else :
        elo_rating = elo_df[elo_df['Game_ID'] == prev_game['Game_ID']]['A_Team_Elo_After'].values[0]
  
    if prev_game['Season'] != season :
        return (0.75 * elo_rating) + (0.25 * 1505)
    else :
        return elo_rating



def add_novel_features(df):
    import warnings
    warnings.filterwarnings("ignore")

    # create avg win pct in last 10 games feature
    for season in df['Season'].unique() :
        season_stats = df[df['Season'] == season].sort_values(by='Date').reset_index(drop=True)
        for index, row in df.iterrows() : 
            game_id = row['Game_ID']
            game_date = row['Date']
            h_team = row['Home']
            a_team = row['Away']
            df.loc[index,'Home_W_Pct_10'] = get_avg_win_pct_last_n_games(h_team, game_date, df, 10)
            df.loc[index,'Away_W_Pct_10'] = get_avg_win_pct_last_n_games(a_team, game_date, df, 10)
    
    # create elo feature
    df.sort_values(by = 'Date', inplace = True)
    df.reset_index(inplace=True, drop = True)
    elo_df = pd.DataFrame(columns=['Game_ID', 'H_Team', 'A_Team', 'H_Team_Elo_Before', 'A_Team_Elo_Before', 'H_Team_Elo_After', 'A_Team_Elo_After'])
    teams_elo_df = pd.DataFrame(columns=['Game_ID','Team', 'Elo', 'Date', 'Where_Played', 'Season']) 

    for index, row in df.iterrows(): 
        game_id = row['Game_ID']
        game_date = row['Date']
        season = row['Season']
        h_team, a_team = row['Home'], row['Away']
        h_score, a_score = row['H_Score'], row['A_Score'] 

        if (h_team not in elo_df['H_Team'].values and h_team not in elo_df['A_Team'].values) :
            h_team_elo_before = 1500
        else :
            h_team_elo_before = get_prev_elo(h_team, game_date, season, df, elo_df)

        if (a_team not in elo_df['H_Team'].values and a_team not in elo_df['A_Team'].values) :
            a_team_elo_before = 1500
        else :
            a_team_elo_before = get_prev_elo(a_team, game_date, season, df, elo_df)

        h_team_elo_after, a_team_elo_after = update_elo(h_score, a_score, h_team_elo_before, a_team_elo_before, 69)

        new_row = {'Game_ID': game_id, 'H_Team': h_team, 'A_Team': a_team, 'H_Team_Elo_Before': h_team_elo_before, 'A_Team_Elo_Before': a_team_elo_before, \
                                                                            'H_Team_Elo_After' : h_team_elo_after, 'A_Team_Elo_After': a_team_elo_after}
        teams_row_one = {'Game_ID': game_id,'Team': h_team, 'Elo': h_team_elo_before, 'Date': game_date, 'Where_Played': 'Home', 'Season': season}
        teams_row_two = {'Game_ID': game_id,'Team': a_team, 'Elo': a_team_elo_before, 'Date': game_date, 'Where_Played': 'Away', 'Season': season}
    
        elo_df.loc[len(elo_df)] = new_row
        teams_elo_df.loc[len(teams_elo_df)] = teams_row_one
        teams_elo_df.loc[len(teams_elo_df)] = teams_row_two

    dates = list(set([d.strftime("%m-%d-%Y") for d in teams_elo_df["Date"]]))
    dates = sorted(dates, key=lambda x: time.strptime(x, '%m-%d-%Y'))
    teams = df["Away"]
    dataset = pd.DataFrame(columns=dates)
    dataset["Team"] = teams.drop_duplicates()
    dataset = dataset.set_index("Team")

    for index, row in teams_elo_df.iterrows():
        date = row["Date"].strftime("%m-%d-%Y")
        team = row["Team"]
        elo = row["Elo"]
        dataset[date][team] = elo

    teams_elo_df['Elo'] = teams_elo_df['Elo'].astype(float)
    df = df.merge(elo_df.drop(columns=['H_Team', 'A_Team']), on ='Game_ID')
    
    return df



def prepare_data(df):
    df = df.drop(['H_Team_Elo_After', 'A_Team_Elo_After'], axis=1)
    df["H_Team_Elo_Before"] = df.H_Team_Elo_Before.astype(float)
    df["A_Team_Elo_Before"] = df.A_Team_Elo_Before.astype(float)
    final_df = df.drop(['Home', 'Away', 'Game_ID', 'H_Score', 'A_Score', 'Date', 'Season'], axis=1)

    X = final_df.drop(columns = 'Result')
    y = final_df['Result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    print(f'X train shape: {X_train.shape}')
    print(f'X test shape: {X_test.shape}')

    return X_train, X_test, y_train, y_test



def eda(df):
    plt.figure(figsize=(25, 25))
    correlation = df[['H_W_PCT', 'H_REB', 'H_AST',
        'H_TOV', 'H_STL', 'H_BLK', 'H_PLUS_MINUS', 'H_OFF_RATING',
        'H_DEF_RATING', 'H_TS_PCT', 'H_Team_Elo_Before', 'Home_W_Pct_10', 'Result']].corr()
    sns.heatmap(correlation, annot=True)
    plt.show()



#script to test the effectivenes of each model, uses default parameters
#test six different classification models 
def run_exps(X_train, X_test, y_train, y_test) :
    '''
    Lightweight script to test many models and find winners
    :param X_train: training split
    :param y_train: training target vector
    :param X_test: test split
    :param y_test: test target vector
    :return: DataFrame of predictions
    '''
    
    dfs = []
    
    models = [
          ('LogReg', LogisticRegression()), 
          ('RF', RandomForestClassifier()),
          ('KNN', KNeighborsClassifier()),
          ('SVM', SVC()), 
          ('GNB', GaussianNB()),
          ('XGB', XGBClassifier())
        ]
    
    results = []
    names = []
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    target_names = ['win', 'loss']
    
    for name, model in models:
        
        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        print(name)
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        results.append(cv_results)
        names.append(name)
        
        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)
        
    return pd.concat(dfs, ignore_index=True)



def boostrap(final):
    bootstraps = []
    for model in list(set(final.model.values)):
        model_df = final.loc[final.model == model]
        bootstrap = model_df.sample(n=30, replace=True)
        bootstraps.append(bootstrap)
    
    return pd.concat(bootstraps, ignore_index=True)



def evaluate_models(bootstrap_df):
    results_long = pd.melt(bootstrap_df,id_vars=['model'],var_name='metrics', value_name='values')
    time_metrics = ['fit_time','score_time'] # fit time metrics

    ## PERFORMANCE METRICS
    results_long_nofit = results_long.loc[~results_long['metrics'].isin(time_metrics)] # get df without fit data
    results_long_nofit = results_long_nofit.sort_values(by='values')

    ## TIME METRICS
    results_long_fit = results_long.loc[results_long['metrics'].isin(time_metrics)] # df with fit data
    results_long_fit = results_long_fit.sort_values(by='values')

    metrics = list(set(results_long_nofit.metrics.values))
    bootstrap_df.groupby(['model'])[metrics].agg([np.std, np.mean])

    return results_long_nofit, results_long_fit

    
def generate_plots(results_long_nofit, results_long_fit):    
    # generate plot comparing models' classification metrics
    plt.figure(figsize=(20, 12))
    sns.set_theme(font_scale=2.5)
    g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_nofit, palette="Set3")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Comparison of Model by Classification Metric')
    plt.savefig('./benchmark_models_performance.png',dpi=300)
    plt.show()
    
    # generate plot comparing models' fit and score time
    plt.figure(figsize=(20, 12))
    sns.set_theme(font_scale=2.5)
    g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_fit, palette="Set3")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Comparison of Model by Fit and Score Time')
    plt.savefig('./benchmark_models_time.png',dpi=300)
    plt.show()



def tune_GNB(X_train, X_test, y_train, y_test):
    nb_classifier = GaussianNB()
    target_names = ['Win', 'Loss']
    params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
    
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
    gs_NB = model_selection.GridSearchCV(estimator=nb_classifier, 
                    param_grid=params_NB, 
                    cv=kfold,   
                    verbose=1, 
                    scoring='accuracy', n_jobs=-1) 

    gs_NB.fit(X_train, y_train)

    best_gs_grid = gs_NB.best_estimator_
    best_gs_grid.fit(X_train, y_train)
    y_pred_best_gs = best_gs_grid.predict(X_test)

    print(classification_report(y_test, y_pred_best_gs, target_names=target_names))
    confusionMatrix = confusion_matrix(y_test, y_pred_best_gs)

    return gs_NB, confusionMatrix



# Saves the model in folder to be used in future
# filename should be end in '.pkl'
def save_model(model, filename):

    with open(filename, 'wb') as file:
        pickle.dump(model, file)



def main():
    df = get_raw_data()
    df = add_novel_features(df)
    eda(df)
    X_train, X_test, y_train, y_test = prepare_data(df)
    final = run_exps(X_train, X_test, y_train, y_test)
    bootstrap_df = boostrap(final)
    results_long_nofit, results_long_fit = evaluate_models(bootstrap_df)
    generate_plots(results_long_nofit, results_long_fit)
    model, confusion_matrix = tune_GNB(X_train, X_test, y_train, y_test)
    save_model(model, 'model')
    
if __name__=="__main__":
    main()