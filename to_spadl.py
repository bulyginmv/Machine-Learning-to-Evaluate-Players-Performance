from io import BytesIO

from pathlib import Path

from tqdm.notebook import tqdm

from urllib.parse import urlparse
from urllib.request import urlopen, urlretrieve

from zipfile import ZipFile, is_zipfile

import pandas as pd  # version 1.0.3

from socceraction.spadl.wyscout import convert_to_spadl

import warnings
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

data_files = {
    'events': 'https://ndownloader.figshare.com/files/14464685',  # ZIP file containing one JSON file for each competition
    'matches': 'https://ndownloader.figshare.com/files/14464622',  # ZIP file containing one JSON file for each competition
    'players': 'https://ndownloader.figshare.com/files/15073721',  # JSON file
    'teams': 'https://ndownloader.figshare.com/files/15073697'  # JSON file
}

for url in tqdm(data_files.values()):
    url_s3 = urlopen(url).geturl()
    path = Path(urlparse(url_s3).path)
    file_name = path.name
    file_local, _ = urlretrieve(url_s3, file_name)
    if is_zipfile(file_local):
        with ZipFile(file_local) as zip_file:
            zip_file.extractall()


def read_json_file(filename):
    with open(filename, 'rb') as json_file:
        return BytesIO(json_file.read()).getvalue().decode('unicode_escape')


json_teams = read_json_file('teams.json')
df_teams = pd.read_json(json_teams)

df_teams.to_hdf('wyscout.h5', key='teams', mode='w')

json_players = read_json_file('players.json')
df_players = pd.read_json(json_players)

df_players.to_hdf('wyscout.h5', key='players', mode='a')

competitions = ['World Cup']

dfs_matches = []
for competition in competitions:
    competition_name = competition.replace(' ', '_')
    file_matches = f'matches_{competition_name}.json'
    json_matches = read_json_file(file_matches)
    df_matches = pd.read_json(json_matches)
    dfs_matches.append(df_matches)
df_matches = pd.concat(dfs_matches)

df_matches.to_hdf('wyscout.h5', key='matches', mode='a')

for competition in competitions:
    competition_name = competition.replace(' ', '_')
    file_events = f'events_{competition_name}.json'
    json_events = read_json_file(file_events)
    df_events = pd.read_json(json_events)
    print(competition)
    df_events_matches = df_events.groupby('matchId', as_index=False)
    for match_id, df_events_match in df_events_matches:
        df_events_match.to_hdf('wyscout.h5', key=f'events/match_{match_id}', mode='a')

convert_to_spadl('wyscout.h5', 'spadl.h5')