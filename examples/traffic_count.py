import pandas as pd
from pathlib import Path

LOG_DIR = Path('../runs/count')
LOG = '0'
LOG = LOG_DIR.joinpath(f'exp-{LOG}.txt')

df = pd.read_csv(LOG)
df.columns = ['frame','direction','zone','track_id']
speed = df[['frame', 'track_id']]
print(speed)

for track in df.track_id.unique():
    direction = df[df.track_id == track]['zone'].unique()
    # direction = df[df.track_id == track][['track_id', 'zone']]

    # if direction.size == 2:
    #     print(track, speed,  direction)
