from tqdm.notebook import tqdm


import pandas as pd  # version 1.0.3

from xgboost import XGBClassifier  # version 1.0.2

import socceraction.classification.features as features
import socceraction.classification.labels as labels

from socceraction.vaep import value

import warnings
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

df_games = pd.read_hdf('spadl.h5', key='games')
df_actiontypes = pd.read_hdf('spadl.h5', key='actiontypes')
df_bodyparts = pd.read_hdf('spadl.h5', key='bodyparts')
df_results = pd.read_hdf('spadl.h5', key='results')

nb_prev_actions = 3

functions_features = [
    features.actiontype_onehot,
    features.bodypart_onehot,
    features.result_onehot,
    features.goalscore,
    features.startlocation,
    features.endlocation,
    features.movement,
    features.space_delta,
    features.startpolar,
    features.endpolar,
    features.team,
    features.time_delta
]

for _, game in tqdm(df_games.iterrows(), total=len(df_games)):
    game_id = game['game_id']
    df_actions = pd.read_hdf('spadl.h5', key=f'actions/game_{game_id}')
    df_actions = (df_actions
                  .merge(df_actiontypes, how='left')
                  .merge(df_results, how='left')
                  .merge(df_bodyparts, how='left')
                  .reset_index(drop=True)
                  )

    dfs_gamestates = features.gamestates(df_actions, nb_prev_actions=nb_prev_actions)
    dfs_gamestates = features.play_left_to_right(dfs_gamestates, game['home_team_id'])

    df_features = pd.concat([function(dfs_gamestates) for function in functions_features], axis=1)
    df_features.to_hdf('features.h5', key=f'game_{game_id}')

functions_labels = [
    labels.scores,
    labels.concedes
]

for _, game in tqdm(df_games.iterrows(), total=len(df_games)):
    game_id = game['game_id']
    df_actions = pd.read_hdf('spadl.h5', key=f'actions/game_{game_id}')
    df_actions = (df_actions
                  .merge(df_actiontypes, how='left')
                  .merge(df_results, how='left')
                  .merge(df_bodyparts, how='left')
                  .reset_index(drop=True)
                  )

    df_labels = pd.concat([function(df_actions) for function in functions_labels], axis=1)
    df_labels.to_hdf('labels.h5', key=f'game_{game_id}')

columns_features = features.feature_column_names(functions_features, nb_prev_actions=nb_prev_actions)

dfs_features = []
for _, game in tqdm(df_games.iterrows(), total=len(df_games)):
    game_id = game['game_id']
    df_features = pd.read_hdf('features.h5', key=f'game_{game_id}')
    dfs_features.append(df_features[columns_features])
df_features = pd.concat(dfs_features).reset_index(drop=True)

columns_labels = [
    'scores',
    'concedes'
]

dfs_labels = []
for _, game in tqdm(df_games.iterrows(), total=len(df_games)):
    game_id = game['game_id']
    df_labels = pd.read_hdf('labels.h5', key=f'game_{game_id}')
    dfs_labels.append(df_labels[columns_labels])
df_labels = pd.concat(dfs_labels).reset_index(drop=True)

models = {}
for column_labels in columns_labels:
    model = XGBClassifier()
    model.fit(df_features, df_labels[column_labels])
    models[column_labels] = model


dfs_predictions = {}
for column_labels in columns_labels:
    model = models[column_labels]
    probabilities = model.predict_proba(df_features)
    predictions = probabilities[:, 1]
    dfs_predictions[column_labels] = pd.Series(predictions)
df_predictions = pd.concat(dfs_predictions, axis=1)

dfs_game_ids = []
for _, game in tqdm(df_games.iterrows(), total=len(df_games)):
    game_id = game['game_id']
    df_actions = pd.read_hdf('spadl.h5', key=f'actions/game_{game_id}')
    dfs_game_ids.append(df_actions['game_id'])
df_game_ids = pd.concat(dfs_game_ids, axis=0).astype('int').reset_index(drop=True)

df_predictions = pd.concat([df_predictions, df_game_ids], axis=1)

df_predictions_per_game = df_predictions.groupby('game_id')

for game_id, df_predictions in tqdm(df_predictions_per_game):
    df_predictions = df_predictions.reset_index(drop=True)
    df_predictions[columns_labels].to_hdf('predictions.h5', key=f'game_{game_id}')

df_players = pd.read_hdf('spadl.h5', key='players')
df_teams = pd.read_hdf('spadl.h5', key='teams')

dfs_values = []
for _, game in tqdm(df_games.iterrows(), total=len(df_games)):
    game_id = game['game_id']
    df_actions = pd.read_hdf('spadl.h5', key=f'actions/game_{game_id}')
    df_actions = (df_actions
                  .merge(df_actiontypes, how='left')
                  .merge(df_results, how='left')
                  .merge(df_bodyparts, how='left')
                  .merge(df_players, how='left')
                  .merge(df_teams, how='left')
                  .reset_index(drop=True)
                  )

    df_predictions = pd.read_hdf('predictions.h5', key=f'game_{game_id}')
    df_values = value(df_actions, df_predictions['scores'], df_predictions['concedes'])

    df_all = pd.concat([df_actions, df_predictions, df_values], axis=1)
    dfs_values.append(df_all)

df_values = (pd.concat(dfs_values)
    .sort_values(['game_id', 'period_id', 'time_seconds'])
    .reset_index(drop=True)
)

pass_val = df_values.loc[df_values.type_name == 'shot']
top = pass_val.sort_values('vaep_value', ascending=False).head(10)

rating_per_game = pd.DataFrame(columns = ['game_id', 'team_name', 'short_name', 'vaep_count', 'vaep_sum', 'minutes','vaep_rating'])
for game in df_values.game_id.unique():
    temp = df_values.loc[df_values.game_id == game]
    df_ranking = (temp[['player_id', 'team_name', 'short_name', 'vaep_value']]
        .groupby(['player_id', 'team_name', 'short_name'])
        .agg(vaep_count=('vaep_value', 'count'), vaep_sum=('vaep_value', 'sum'))
        .sort_values('vaep_sum', ascending=False)
        .reset_index()
                 )
    rating_per_game = rating_per_game.append(df_ranking.iloc[0], ignore_index=True)



#Normanlize to 90 mins

df_player_games = pd.read_hdf('spadl.h5', 'player_games')
df_player_games = df_player_games[df_player_games['game_id'].isin(df_games['game_id'])]

rating = []
time = []
for index, row in rating_per_game.iterrows():
    minutes = df_player_games.loc[(df_player_games['game_id']==int(row['game_id'])) &
                                   (df_player_games['player_id']==int(row['player_id'])) , ['minutes_played']]
    time.append(minutes.iloc[0]['minutes_played'])

rating_per_game['minutes']=time

rating_per_game = rating_per_game.sort_values('vaep_sum', ascending=True)

top = rating_per_game.tail(10)

low = rating_per_game.head(10)

df_player_games.loc[(df_player_games['game_id']==int(2058017.0)) & (df_player_games['player_id']==int(69616)) , ['minutes_played']]

df_minutes_played = (df_player_games[['game_id','player_id', 'minutes_played']]
    .groupby('player_id')
    .sum()
    .reset_index()
)

df_ranking_p90 = df_ranking.merge(df_minutes_played)
df_ranking_p90 = df_ranking_p90[df_ranking_p90['minutes_played'] > 360]
df_ranking_p90['vaep_rating'] = df_ranking_p90['vaep_sum'] * 90 / df_ranking_p90['minutes_played']
df_ranking_p90 = df_ranking_p90.sort_values('vaep_rating', ascending=False)

df_ranking_p90 = rating_per_game.sort_values('vaep_rating', ascending=False)
df_ranking_p90.head(10)

top_plot = pd.DataFrame (columns = ['Name','rating'])

name = []
rating = []
for i in range (10):
    name.append(low.iloc[i]['short_name']+' (' +
       df_games[df_games['game_id']==low.iloc[i]['game_id']].iloc[0]['home_team_name']+ ' '+
      'vs '+df_games[df_games['game_id']==low.iloc[i]['game_id']].iloc[0]['away_team_name']+')')
    rating.append(low.iloc[i]['vaep_sum'],)

top_plot['Name']=name
top_plot['rating']= rating