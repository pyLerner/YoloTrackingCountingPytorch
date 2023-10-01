import pandas as pd
from pathlib import Path

LOG_DIR = Path('/home/pyler/netdisk/Projects/YoloTrackingCountingPytorch/runs/count')
# print(LOG_DIR)

LOG = LOG_DIR.rglob('*.txt') # -> map object (all logs)
# print(list(LOG))
LOG = max(map(lambda x: x.stem[-3:], LOG)) # -> str
# print(LOG, type(LOG))
# LOG = '001'
LOG = LOG_DIR.joinpath(f'exp-{LOG}.txt') # -> log Path object

df = pd.read_csv(LOG)
df.columns = ['frame','direction','zone','track']

direction_df = pd.DataFrame((), columns=['car', 'zone_in', 'zone_out', 'delay'])

for car in df.track.unique():
    start = df[(df.direction == 0) & (df.track == car)]['frame'].min()
    stop = df[(df.direction == 1) & (df.track == car)]['frame'].min()
    # print(car, 'frames', start, stop)
    delay = round((stop - start) / 25, 1) # seconds (FPS = 25)
    
    zone_in = df[(df.track == car) & (df.direction == 0)]['zone'].min()
    zone_out = df[(df.track == car) & (df.direction == 1)]['zone'].min()
    # print(car, 'zones', zone_in, zone_out)
    
    seria = pd.DataFrame({'car': [car], 'zone_in': [zone_in], 'zone_out': [zone_out], 'delay': [delay]})
    seria.dropna(inplace=True)
    
    direction_df = pd.concat([seria, direction_df], ignore_index=True)
       
    # direction_df.dropna(inplace=True)  # Отсев ложных детекций
    direction_df = direction_df[direction_df.delay > .6]  # Отсев ложных маневров, у которых время проезда не больше 1 сек.
    # direction_df = direction_df[direction_df.zone_in != direction_df.zone_out]
    
# print(direction_df)

direction_types = {'From 0 to 1': 'налево',
                   'From 0 to 2': 'прямо',
                   'From 0 to 3': 'направо',
                   'From 1 to 0': 'направо',
                   'From 1 to 2': 'налево',
                   'From 1 to 3': 'прямо',
                   'From 2 to 0': 'прямо',
                   'From 2 to 1': 'направо',
                   'From 2 to 3': 'налево',
                   'From 3 to 0': 'налево',
                   'From 3 to 1': 'прямо',
                   'From 3 to 2': 'направо',
                   }
# directions = list(direction_types.keys())
# direction_types = {
#     'stright': [x for x in direction_types.keys() if direction_types[x] == 'прямо'],
#     'left': [x for x in direction_types.keys() if direction_types[x] == 'налево'],
#     'right': [x for x in direction_types.keys() if direction_types[x] == 'направо'],
# }

# print(direction_types)

total_df = pd.DataFrame()


for zone_in in direction_df.zone_in.unique():
    for zone_out in direction_df.zone_out.unique():
        if zone_in != zone_out:
            street = f'From {zone_in} to {zone_out}'
            move_type = direction_types[street]
            moves = direction_df[
                (direction_df.zone_in == zone_in) & (direction_df.zone_out == zone_out)
            ]
        
        if moves.shape[0]:
            amount_moves = moves.shape[0] 

        mean_delay = round(moves.delay.mean(), 1)

        out = {'Street': [street], 
               'Move_type': [move_type],
               'Amount_Moves': [amount_moves],
               'Mean_Delay': [mean_delay]}
        
        out = pd.DataFrame(out)
        # print(out)
        

        total_df = pd.concat([out, total_df], ignore_index=True)
        # print(total_df)

# TODO: Проверить откуда берутся дубликаты строк        
total_df.drop_duplicates(keep='first', inplace=True)
total_df.sort_values(['Move_type', 'Street'], ignore_index=True, inplace=True)
total_df.columns=['Участок дороги', 
                  'Направление движения',
                  'Количество ТС,',
                  'Средняя задержка, сек.']
print(total_df)

total_df.to_html('result.html', index=False)

        # res = direction_df[(direction_df.zone_in == zone_in) & (direction_df.zone_out == zone_out)]
        # if res.shape[0]:
        #     print(f'From {zone_in} to {zone_out} - {res.shape[0]} with mean delay {res.delay.mean():.1f} sec')


# print(total_df.groupby(['Street', 'Move_type'])['Amount_Moves'].count())

# print(total_df[total_df.Move_type == 'прямо'].sort_values('Street'))