import pandas as pd
import numpy as np
import xG_model
import os
import gc
import json

def get_shots_data(df, season=2022):
    # create Play_Id field for joining later
    df = df.reset_index()
    df.rename({'index':'Play_Id'}, axis=1, inplace=True)

    # add a couple fields
    df['Season'] = season
    df['Empty_Net'] = (df['Away_Goalie'].isnull()) | (df['Home_Goalie'].isnull())

    # fix teams
    df['Ev_Team'] = df['Ev_Team'].replace({'PHX':'ARI', 'S.J':'SJS', 'L.A':'LAK', 'T.B':'TBL', 'N.J':'NJD'})
    df['Home_Team'] = df['Home_Team'].replace({'PHX':'ARI', 'S.J':'SJS', 'L.A':'LAK', 'T.B':'TBL', 'N.J':'NJD'})
    df['Away_Team'] = df['Away_Team'].replace({'PHX':'ARI', 'S.J':'SJS', 'L.A':'LAK', 'T.B':'TBL', 'N.J':'NJD'})

    # to add xG values, filter for only specific events that are relevant for xG
    shots = df.loc[df['Event'].isin(['MISS','SHOT','GOAL','FAC','HIT','BLOCK','GIVE','TAKE'])]

    # remove null location data (this is a key part of xG)
    shots = shots.loc[~shots['xC'].isnull()]
    shots = shots.loc[~shots['yC'].isnull()]

    # get previous event time and location
    shots = shots.sort_values(by=['Game_Id','Period','Seconds_Elapsed'])
    shots['prev_Game_Id'] = shots['Game_Id'].shift(1)
    shots['prev_Period'] = shots['Period'].shift(1)
    shots['keepPrev'] = ((shots['prev_Game_Id']==shots['Game_Id']) & (shots['prev_Period']==shots['Period'])).astype(int)
    shots['prev_Event'] = shots['Event'].shift(1)
    shots['prev_Seconds_Elapsed'] = shots['Seconds_Elapsed'].shift(1)
    shots['prev_xC'] = shots['xC'].shift(1)
    shots['prev_yC'] = shots['yC'].shift(1)
    shots['prev_Ev_Team'] = shots['Ev_Team'].shift(1)
    shots['prev_sameTeam'] = (shots['prev_Ev_Team']==shots['Ev_Team']).astype(int)
    shots.at[shots['keepPrev']==0, ['prev_Event']] = np.NaN
    shots.at[shots['keepPrev']==0, ['prev_Seconds_Elapsed','prev_xC','prev_yC']] = 0.

    # get time elapsed, and distance from previous event
    shots['timeSincePrev'] = shots['Seconds_Elapsed'] - shots['prev_Seconds_Elapsed']
    shots['distanceSincePrev'] = np.sqrt(np.square(shots['xC']-shots['prev_xC']) + np.square(shots['yC']-shots['prev_yC']))
    shots['yDistanceSincePrev'] = np.abs(shots['yC'] - shots['prev_yC'])
    shots['xDistanceSincePrev'] = np.abs(shots['xC'] - shots['prev_xC'])

    # filter for only events we want to predict xG on (other events are only relevant as previous events before the attempt)
    shots = shots.loc[shots['Event'].isin(['MISS','SHOT','GOAL'])]

    # filter for only most common game states
    shots = shots.loc[shots['Strength'].isin(['5x5','4x5','3x5','5x4','4x4','5x3','4x3','3x4','3x3'])]

    # field for whether the shooting team has an empty net to shoot at
    shots['ShotIntoEmptyNet'] = (((shots['Ev_Team']==shots['Home_Team']) & shots['Away_Goalie'].isnull()) | \
        ((shots['Ev_Team']==shots['Away_Team']) & shots['Home_Goalie'].isnull())).astype(int)

    # remove empty net shots
    shots = shots.loc[shots['ShotIntoEmptyNet']==0]

    # field for rebound and rush shots
    shots['ShotCategory'] = 'Unknown'
    shots.loc[(shots['timeSincePrev']<=2)&(shots['prev_sameTeam']==1)&(shots['prev_Event']=='SHOT'), 'ShotCategory'] = 'Rebound'
    shots.loc[(shots['timeSincePrev']<=4)&(shots['xDistanceSincePrev']>=50), 'ShotCategory'] = 'Rush'

    # get previous shot time and location, and then calculate derived metrics
    shots['prevShot_Game_Id'] = shots['Game_Id'].shift(1)
    shots['prevShot_Period'] = shots['Period'].shift(1)
    shots['keepPrevShot'] = ((shots['prevShot_Game_Id']==shots['Game_Id']) & (shots['prevShot_Period']==shots['Period'])).astype(int)
    shots['prevShot_Seconds_Elapsed'] = shots['Seconds_Elapsed'].shift(1)
    shots['prevShot_xC'] = shots['xC'].shift(1)
    shots['prevShot_yC'] = shots['yC'].shift(1)
    shots['prevShot_Ev_Team'] = shots['Ev_Team'].shift(1)
    shots['prevShot_sameTeam'] = (shots['prevShot_Ev_Team']==shots['Ev_Team']).astype(int)
    shots.at[shots['keepPrevShot']==0, ['prevShot_Seconds_Elapsed','prevShot_xC','prevShot_yC','prevShot_Ev_Team']] = np.NaN
    shots['timeSincePrevShot'] = shots['Seconds_Elapsed'] - shots['prevShot_Seconds_Elapsed']
    shots['distanceSincePrevShot'] = np.sqrt(np.square(shots['xC']-shots['prevShot_xC']) + np.square(shots['yC']-shots['prevShot_yC']))
    shots['yDistanceSincePrevShot'] = np.abs(shots['yC'] - shots['prevShot_yC'])

    # adjust shot locations so everything is on the same side of the ice
    # TODO this isn't quite right - shots from the D zone (into an empty net, for example), will not be adjusted correctly
    shots['loc_adjust_factor'] = (((shots['xC']>0).astype(int).astype(float)) - 0.5) * 2
    shots['x_adj'] = shots['xC']*shots['loc_adjust_factor']
    shots['y_adj'] = shots['yC']*shots['loc_adjust_factor']
    shots['prev_loc_adjust_factor'] = (((shots['prev_xC']>0).astype(int).astype(float)) - 0.5) * 2
    shots['prev_x_adj'] = shots['prev_xC']*shots['prev_loc_adjust_factor']
    shots['prev_y_adj'] = shots['prev_yC']*shots['prev_loc_adjust_factor']
    shots['prevShot_loc_adjust_factor'] = (((shots['prevShot_xC']>0).astype(int).astype(float)) - 0.5) * 2
    shots['prevShot_x_adj'] = shots['prevShot_xC']*shots['prevShot_loc_adjust_factor']
    shots['prevShot_y_adj'] = shots['prevShot_yC']*shots['prevShot_loc_adjust_factor']

    # fill nulls
    shots[['prevShot_Seconds_Elapsed','prevShot_yC','prevShot_y_adj','prevShot_xC']] = \
        shots[['prevShot_Seconds_Elapsed','prevShot_yC','prevShot_y_adj','prevShot_xC']].fillna(0)
    shots[['prevShot_x_adj','distanceSincePrevShot','yDistanceSincePrevShot']] = \
        shots[['prevShot_x_adj','distanceSincePrevShot','yDistanceSincePrevShot']].fillna(-1)
    shots[['timeSincePrevShot']] = shots[['timeSincePrevShot']].fillna(1200)

    # fix time since prev shot if prev shot was in another period
    shots.at[shots['timeSincePrevShot']<0, 'timeSincePrevShot'] = 1200

    # adjust score to be score for and against, instead of home and away
    shots['homeTeamShot'] = (shots['Home_Team']==shots['Ev_Team']).astype(int)
    shots['scoreFor'] = (shots['Home_Score']*shots['homeTeamShot']) + (shots['Away_Score']*(1-shots['homeTeamShot']))
    shots['scoreAgainst'] = (shots['Away_Score']*shots['homeTeamShot']) + (shots['Home_Score']*(1-shots['homeTeamShot']))
    shots['scoreDiff'] = shots['scoreFor'] - shots['scoreAgainst']

    # reverse strength for away team, so that 5x4 always means PP and 4x5 always means PK
    shots.loc[shots['Ev_Team']==shots['Away_Team'], 'Strength'] = shots.loc[shots['Ev_Team']==shots['Away_Team'], 'Strength'].str[::-1]

    # add target variable
    shots['goal'] = (shots['Event']=='GOAL').astype(int)

    return df, shots

def add_xG_to_pbp(df, shots):
    # make xG predictions
    model = xG_model()
    shots['xG'] = model.predict(shots)
    del model

    # join to original dataframe with all events
    shots = shots[['Play_Id','xG','ShotCategory']]
    df = df.merge(shots, on='Play_Id', how='left')
    del shots

    # downcast some fields
    df['Period'] = df['Period'].astype(np.int8)
    df['Seconds_Elapsed'] = df['Seconds_Elapsed'].astype(np.int16)
    df['p1_ID'] = df['p1_ID'].fillna(-1).astype(np.int32)
    df['p2_ID'] = df['p2_ID'].fillna(-1).astype(np.int32)
    df['p3_ID'] = df['p3_ID'].fillna(-1).astype(np.int32)
    df['awayPlayer1_id'] = df['awayPlayer1_id'].fillna(-1).astype(np.int32)
    df['awayPlayer2_id'] = df['awayPlayer2_id'].fillna(-1).astype(np.int32)
    df['awayPlayer3_id'] = df['awayPlayer3_id'].fillna(-1).astype(np.int32)
    df['awayPlayer4_id'] = df['awayPlayer4_id'].fillna(-1).astype(np.int32)
    df['awayPlayer5_id'] = df['awayPlayer5_id'].fillna(-1).astype(np.int32)
    df['awayPlayer6_id'] = df['awayPlayer6_id'].fillna(-1).astype(np.int32)
    df['homePlayer1_id'] = df['homePlayer1_id'].fillna(-1).astype(np.int32)
    df['homePlayer2_id'] = df['homePlayer2_id'].fillna(-1).astype(np.int32)
    df['homePlayer3_id'] = df['homePlayer3_id'].fillna(-1).astype(np.int32)
    df['homePlayer4_id'] = df['homePlayer4_id'].fillna(-1).astype(np.int32)
    df['homePlayer5_id'] = df['homePlayer5_id'].fillna(-1).astype(np.int32)
    df['homePlayer6_id'] = df['homePlayer6_id'].fillna(-1).astype(np.int32)
    df['Away_Players'] = df['Away_Players'].astype(np.int8)
    df['Home_Players'] = df['Home_Players'].astype(np.int8)
    df['Away_Score'] = df['Away_Score'].astype(np.int8)
    df['Home_Score'] = df['Home_Score'].astype(np.int8)
    df['Away_Goalie_Id'] = df['Away_Goalie_Id'].fillna(-1).astype(np.int32)
    df['Home_Goalie_Id'] = df['Home_Goalie_Id'].fillna(-1).astype(np.int32)
    df['Season'] = df['Season'].astype(np.int32)

    # remove Play_Id field, it's now useless
    df = df.drop(columns=['Play_Id'])

    # xG flurry adjustment
    df = df.sort_values(by=['Date','Game_Id','Period','Seconds_Elapsed'])
    df['prev_xG'] = df.groupby(['Game_Id','Date','Season','Period'])['xG'].shift(1)
    df['xG_flurry'] = df['xG']
    df.loc[df['ShotCategory']=='Rebound', 'xG_flurry'] = df.loc[df['ShotCategory']=='Rebound', 'xG']*(1-df.loc[df['ShotCategory']=='Rebound', 'prev_xG'])
    df = df.drop('prev_xG',1)

    return df

def aggregate_player_data(plays, shifts, homeaway_adjustments='data/score_homeaway_adjustments.csv'):

    # ensure gameid is an int
    shifts['Game_Id'] = shifts['Game_Id'].astype(np.int64)
    plays['Game_Id'] = plays['Game_Id'].astype(np.int64)

    # process one chunk of games at a time
    chunk_size = 10
    gameids = shifts['Game_Id'].unique()
    for i in range(math.ceil(len(gameids)/chunk_size)):
        gameids_chunk = gameids[i*chunk_size:(i+1)*chunk_size]

        pbp_merge = plays.loc[plays['Game_Id'].isin(gameids_chunk)]
        pbp_merge = pbp_merge.loc[~pbp_merge['Home_Zone'].isna(), ['Game_Id','Period','Strength','Seconds_Elapsed','Home_Team','Away_Team','Event','Home_Zone']]
        pbp_merge = pbp_merge.sort_values(by=['Game_Id','Period','Seconds_Elapsed'])
        pbp_merge['eventRank'] = pbp_merge.groupby(['Game_Id','Period','Seconds_Elapsed'])['Event'].rank('first')
        pbp_merge = pbp_merge.loc[pbp_merge['eventRank']==1]
        pbp_merge = pbp_merge.drop('eventRank', 1)
        pbp_merge['Prev_Strength'] = pbp_merge['Strength'].shift(1)
        pbp_merge['Next_Strength'] = pbp_merge['Strength'].shift(-1)
        pbp_merge.loc[pbp_merge['Event']=='PENL', 'Strength'] = pbp_merge.loc[pbp_merge['Event']=='PENL', 'Next_Strength']
        pbp_merge = pbp_merge.loc[(pbp_merge['Strength']!=pbp_merge['Prev_Strength'])|((pbp_merge['Period']==1)&(pbp_merge['Seconds_Elapsed']==0))]
        pbp_merge['End_Seconds_Elapsed'] = pbp_merge['Seconds_Elapsed'].shift(-1)
        pbp_merge['Next_Period'] = pbp_merge['Period'].shift(-1)
        pbp_merge.loc[pbp_merge['End_Seconds_Elapsed'].isna(), 'End_Seconds_Elapsed'] = 1200.
        pbp_merge.loc[(pbp_merge['End_Seconds_Elapsed']==1200)&(pbp_merge['Period']==4)&(pbp_merge['Strength']=='3x3'), 'End_Seconds_Elapsed'] = 300.
        pbp_merge.loc[(pbp_merge['Period']==4)&(pbp_merge['Strength']=='3x3')&(pbp_merge['Next_Period']!=pbp_merge['Period']), 'End_Seconds_Elapsed'] = 300.

        shifts_chunk = shifts.loc[shifts['Game_Id'].isin(gameids_chunk)]
        shifts_chunk = shifts_chunk.merge(shifts_chunk, on=['Game_Id','Period','Date'])
        shifts_chunk = shifts_chunk.loc[((shifts_chunk['Start_y']>=shifts_chunk['Start_x'])&(shifts_chunk['Start_y']<shifts_chunk['End_x']))|\
            ((shifts_chunk['End_y']>shifts_chunk['Start_x'])&(shifts_chunk['End_y']<=shifts_chunk['End_x']))|\
            ((shifts_chunk['End_y']>=shifts_chunk['End_x'])&(shifts_chunk['Start_y']<=shifts_chunk['Start_x']))]
        shifts_chunk = shifts_chunk.loc[shifts_chunk['Player_Id_x']!=shifts_chunk['Player_Id_y']]
        shifts_chunk = shifts_chunk.merge(pbp_merge, on=['Game_Id'])
        shifts_chunk['Period_x'] = shifts_chunk['Period_x'].astype(np.int32)
        shifts_chunk['Period_y'] = shifts_chunk['Period_y'].astype(np.int32)
        shifts_chunk.loc[(shifts_chunk['Seconds_Elapsed']>shifts_chunk['End_Seconds_Elapsed'])&(shifts_chunk['Period_x']==shifts_chunk['Period_y']),
            'End_Seconds_Elapsed'] = 1200.
        shifts_chunk.loc[(shifts_chunk['Seconds_Elapsed']>shifts_chunk['End_Seconds_Elapsed'])&(shifts_chunk['Period_x']>shifts_chunk['Period_y']),
            ['Seconds_Elapsed','Period_y']] = 0.
        shifts_chunk.loc[shifts_chunk['Period_y']==0, 'Period_y'] = shifts_chunk.loc[shifts_chunk['Period_y']==0, 'Period_x']
        shifts_chunk = shifts_chunk.loc[shifts_chunk['Period_x']==shifts_chunk['Period_y']]
        shifts_chunk = shifts_chunk.loc[((shifts_chunk['Seconds_Elapsed']>=shifts_chunk['Start_x'])&(shifts_chunk['Seconds_Elapsed']<shifts_chunk['End_x']))|\
            ((shifts_chunk['End_Seconds_Elapsed']>shifts_chunk['Start_x'])&(shifts_chunk['End_Seconds_Elapsed']<=shifts_chunk['End_x']))|\
            ((shifts_chunk['End_Seconds_Elapsed']>=shifts_chunk['End_x'])&(shifts_chunk['Seconds_Elapsed']<=shifts_chunk['Start_x']))]
        shifts_chunk = shifts_chunk.loc[((shifts_chunk['Seconds_Elapsed']>=shifts_chunk['Start_y'])&(shifts_chunk['Seconds_Elapsed']<shifts_chunk['End_y']))|\
            ((shifts_chunk['End_Seconds_Elapsed']>shifts_chunk['Start_y'])&(shifts_chunk['End_Seconds_Elapsed']<=shifts_chunk['End_y']))|\
            ((shifts_chunk['End_Seconds_Elapsed']>=shifts_chunk['End_y'])&(shifts_chunk['Seconds_Elapsed']<=shifts_chunk['Start_y']))]
        shifts_chunk.loc[shifts_chunk['Team_x']!=shifts_chunk['Home_Team'], 'Strength'] = \
            shifts_chunk.loc[shifts_chunk['Team_x']!=shifts_chunk['Home_Team'], 'Strength'].str[::-1]
        shifts_chunk['Overlap'] = shifts_chunk[['End_x','End_y','End_Seconds_Elapsed']].min(axis=1) - shifts_chunk[['Start_x','Start_y','Seconds_Elapsed']].max(axis=1)
        shifts_chunk['SameTeam'] = shifts_chunk['Team_x']==shifts_chunk['Team_y']

        toi_chunk = shifts_chunk[['Game_Id','Period_x','Team_x','Player_x','Player_Id_x','Start_x','End_x','Date','Strength',
            'Seconds_Elapsed','End_Seconds_Elapsed']].drop_duplicates()
        toi_chunk.columns = ['Game_Id','Period','Team','Player','Player_Id','Start','End','Date','Strength','Seconds_Elapsed','End_Seconds_Elapsed']
        toi_chunk['TOI_5v5'] = toi_chunk.loc[toi_chunk['Strength']=='5x5', ['End','End_Seconds_Elapsed']].min(axis=1) -\
            toi_chunk.loc[toi_chunk['Strength']=='5x5', ['Start','Seconds_Elapsed']].max(axis=1)
        toi_chunk['TOI_PP'] = toi_chunk.loc[toi_chunk['Strength'].isin(['5x4','4x3','5x3']), ['End','End_Seconds_Elapsed']].min(axis=1) -\
            toi_chunk.loc[toi_chunk['Strength'].isin(['5x4','4x3','5x3']), ['Start','Seconds_Elapsed']].max(axis=1)
        toi_chunk['TOI_PK'] = toi_chunk.loc[toi_chunk['Strength'].isin(['4x5','3x4','3x5']), ['End','End_Seconds_Elapsed']].min(axis=1) -\
            toi_chunk.loc[toi_chunk['Strength'].isin(['4x5','3x4','3x5']), ['Start','Seconds_Elapsed']].max(axis=1)
        toi_chunk['TOI'] = toi_chunk[['End','End_Seconds_Elapsed']].min(axis=1) - toi_chunk[['Start','Seconds_Elapsed']].max(axis=1)
        toi_chunk = toi_chunk[['Game_Id','Player','Player_Id','Date','TOI','TOI_5v5','TOI_PP','TOI_PK']]\
            .groupby(['Game_Id','Player','Player_Id','Date'], as_index=False).sum()

        shifts_chunk = shifts_chunk[['Game_Id','Player_x','Player_Id_x','Player_y','Player_Id_y','Strength','SameTeam','Overlap']]
        shifts_chunk = shifts_chunk.groupby(['Game_Id','Player_x','Player_Id_x','Player_y','Player_Id_y','Strength','SameTeam'], as_index=False).sum()
        if i==0:
            toi_overlap = shifts_chunk.copy(deep=True)
            toi = toi_chunk.copy(deep=True)
        else:
            toi_overlap = pd.concat([toi_overlap, shifts_chunk], ignore_index=True)
            toi = pd.concat([toi, toi_chunk], ignore_index=True)

        gc.collect()

        #TODO: consolidate with above? some of this code is the same
        shifts_chunk = shifts.loc[shifts['Game_Id'].isin(gameids_chunk)]
        pbp_merge = plays.loc[plays['Game_Id'].isin(gameids_chunk)]
        pbp_merge = pbp_merge.loc[~pbp_merge['Home_Zone'].isna(), ['Game_Id','Period','Strength','Seconds_Elapsed','Home_Team','Away_Team','Event','Home_Zone']]
        pbp_merge = pbp_merge.sort_values(by=['Game_Id','Period','Seconds_Elapsed'])
        pbp_merge['eventRank'] = pbp_merge.groupby(['Game_Id','Period','Seconds_Elapsed'])['Event'].rank('first')
        pbp_merge = pbp_merge.loc[pbp_merge['eventRank']==1]
        pbp_merge = pbp_merge.drop('eventRank', 1)
        pbp_merge['Prev_Strength'] = pbp_merge['Strength'].shift(1)
        pbp_merge['Next_Strength'] = pbp_merge['Strength'].shift(-1)
        pbp_merge.loc[pbp_merge['Event']=='PENL', 'Strength'] = pbp_merge.loc[pbp_merge['Event']=='PENL', 'Next_Strength']
        pbp_merge = pbp_merge.loc[(pbp_merge['Strength']!=pbp_merge['Prev_Strength'])|((pbp_merge['Period']==1)&(pbp_merge['Seconds_Elapsed']==0))]
        pbp_merge['End_Seconds_Elapsed'] = pbp_merge['Seconds_Elapsed'].shift(-1)
        pbp_merge['Next_Period'] = pbp_merge['Period'].shift(-1)
        pbp_merge.loc[pbp_merge['End_Seconds_Elapsed'].isna(), 'End_Seconds_Elapsed'] = 1200.
        pbp_merge.loc[(pbp_merge['End_Seconds_Elapsed']==1200)&(pbp_merge['Period']==4)&(pbp_merge['Strength']=='3x3'), 'End_Seconds_Elapsed'] = 300.
        pbp_merge.loc[(pbp_merge['Period']==4)&(pbp_merge['Strength']=='3x3')&(pbp_merge['Next_Period']!=pbp_merge['Period']), 'End_Seconds_Elapsed'] = 300.
        shifts_chunk = shifts_chunk.merge(pbp_merge, on=['Game_Id'], how='left')
        shifts_chunk = shifts_chunk.loc[((shifts_chunk['Start']<=shifts_chunk['Seconds_Elapsed'])&\
            (shifts_chunk['End']>=shifts_chunk['Seconds_Elapsed']))|(shifts_chunk['Seconds_Elapsed'].isnull())]

        #reverse Strength value for away team players
        shifts_chunk.loc[shifts_chunk['Team']==shifts_chunk['Away_Team'], 'Strength'] = \
            shifts_chunk.loc[shifts_chunk['Team']==shifts_chunk['Away_Team'], 'Strength'].str[::-1]

        #add zone for 5v5 faceoffs
        shifts_chunk['Zone'] = np.nan
        shifts_chunk.loc[(shifts_chunk['Team']==shifts_chunk['Home_Team'])&(shifts_chunk['Home_Zone']=='Off')&\
            (shifts_chunk['Event']=='FAC')&(shifts_chunk['Strength']=='5x5'), 'Zone'] = 'O'
        shifts_chunk.loc[(shifts_chunk['Team']==shifts_chunk['Home_Team'])&(shifts_chunk['Home_Zone']=='Def')&\
            (shifts_chunk['Event']=='FAC')&(shifts_chunk['Strength']=='5x5'), 'Zone'] = 'D'
        shifts_chunk.loc[(shifts_chunk['Team']==shifts_chunk['Away_Team'])&(shifts_chunk['Home_Zone']=='Def')&\
            (shifts_chunk['Event']=='FAC')&(shifts_chunk['Strength']=='5x5'), 'Zone'] = 'O'
        shifts_chunk.loc[(shifts_chunk['Team']==shifts_chunk['Away_Team'])&(shifts_chunk['Home_Zone']=='Off')&\
            (shifts_chunk['Event']=='FAC')&(shifts_chunk['Strength']=='5x5'), 'Zone'] = 'D'
        shifts_chunk.loc[(shifts_chunk['Home_Zone']=='Neu')&(shifts_chunk['Event']=='FAC')&(shifts_chunk['Strength']=='5x5'), 'Zone'] = 'N'

        #construct zone_starts data
        zone_starts_chunk = shifts_chunk.loc[(shifts_chunk['Event']=='FAC')&(shifts_chunk['Strength']=='5x5')&\
            (shifts_chunk['Zone']=='D'), ['Game_Id','Date','Player','Player_Id','Zone','Team']].groupby(\
            ['Game_Id','Date','Player','Player_Id','Zone']).agg({'Team':'count'})
        zone_starts_chunk.columns = ['DZoneStartCount_5v5']
        zone_starts_chunk = zone_starts_chunk.reset_index()
        zone_starts_chunk = zone_starts_chunk.drop('Zone',1)

        nzone_starts = shifts_chunk.loc[(shifts_chunk['Event']=='FAC')&(shifts_chunk['Strength']=='5x5')&\
            (shifts_chunk['Zone']=='N'), ['Game_Id','Date','Player','Player_Id','Zone','Team']].groupby(\
            ['Game_Id','Date','Player','Player_Id','Zone']).agg({'Team':'count'})
        nzone_starts.columns = ['NZoneStartCount_5v5']
        nzone_starts = nzone_starts.reset_index()
        nzone_starts = nzone_starts.drop('Zone',1)
        zone_starts_chunk = zone_starts_chunk.merge(nzone_starts, on=['Game_Id','Date','Player','Player_Id'], how='outer')

        ozone_starts = shifts_chunk.loc[(shifts_chunk['Event']=='FAC')&(shifts_chunk['Strength']=='5x5')&\
            (shifts_chunk['Zone']=='O'), ['Game_Id','Date','Player','Player_Id','Zone','Team']].groupby(\
            ['Game_Id','Date','Player','Player_Id','Zone']).agg({'Team':'count'})
        ozone_starts.columns = ['OZoneStartCount_5v5']
        ozone_starts = ozone_starts.reset_index()
        ozone_starts = ozone_starts.drop('Zone',1)
        zone_starts_chunk = zone_starts_chunk.merge(ozone_starts, on=['Game_Id','Date','Player','Player_Id'], how='outer')

        if i==0:
            zone_starts = zone_starts_chunk.copy(deep=True)
        else:
            zone_starts = pd.concat([zone_starts, zone_starts_chunk], ignore_index=True)

        gc.collect()

    # read adjustments data
    adjustments = pd.read_csv(homeaway_adjustments)

    # calculate some stats in the disaggregated data, to sum later
    plays['Goals'] = ((plays['Event'] == 'GOAL') & (plays['Strength']!='0x0')).astype(np.int16)
    plays['Shootout_Goals'] = ((plays['Event'] == 'GOAL') & (plays['Strength']=='0x0')).astype(np.int16)
    plays['Shots'] = ((plays['Event'].isin(['SHOT','GOAL'])) & (plays['Strength']!='0x0')).astype(np.int16)
    plays['ShotAttempts'] = ((plays['Event'].isin(['SHOT','MISS','GOAL','BLOCK'])) & (plays['Strength']!='0x0')).astype(np.int16)
    plays['UnblockedShotAttempts'] = ((plays['Event'].isin(['SHOT','MISS','GOAL'])) & (plays['Strength']!='0x0')).astype(np.int16)
    plays['Goals_5v5'] = ((plays['Event'] == 'GOAL') & (plays['Strength'].isin(['5x5'])) & (~plays['Empty_Net'])).astype(np.int16)
    plays['Shots_5v5'] = ((plays['Event'].isin(['SHOT','GOAL'])) & (plays['Strength'].isin(['5x5'])) & (~plays['Empty_Net'])).astype(np.int16)
    plays['ShotAttempts_5v5'] = ((plays['Event'].isin(['SHOT','MISS','GOAL','BLOCK'])) & (plays['Strength'].isin(['5x5'])) & (~plays['Empty_Net'])).astype(np.int16)
    plays['UnblockedShotAttempts_5v5'] = ((plays['Event'].isin(['SHOT','MISS','GOAL'])) & (plays['Strength'].isin(['5x5'])) & (~plays['Empty_Net'])).astype(np.int16)
    plays['xG_5v5'] = np.nan
    plays.loc[plays['Strength'].isin(['5x5']), 'xG_5v5'] = plays.loc[(plays['Strength'].isin(['5x5'])) & (~plays['Empty_Net'])]['xG']
    plays['xG_flurry_5v5'] = np.nan
    plays.loc[plays['Strength'].isin(['5x5']), 'xG_flurry_5v5'] = plays.loc[(plays['Strength'].isin(['5x5'])) & (~plays['Empty_Net'])]['xG_flurry']
    plays['Penalties'] = ((plays['Event']=='PENL')&(~(plays['Type'].str.contains('Fight')).fillna(False))).astype(np.int16)

    #reverse strength for away team
    plays.loc[plays['Ev_Team']==plays['Away_Team'], 'Strength'] = plays.loc[plays['Ev_Team']==plays['Away_Team'], 'Strength'].str[::-1]

    plays['Goals_PP'] = ((plays['Event'] == 'GOAL') & (plays['Strength'].isin(['5x4','5x3','4x3']))).astype(np.int16)
    plays['Shots_PP'] = ((plays['Event'].isin(['SHOT','GOAL'])) & (plays['Strength'].isin(['5x4','5x3','4x3']))).astype(np.int16)
    plays['ShotAttempts_PP'] = ((plays['Event'].isin(['SHOT','MISS','GOAL','BLOCK'])) & (plays['Strength'].isin(['5x4','5x3','4x3']))).astype(np.int16)
    plays['UnblockedShotAttempts_PP'] = ((plays['Event'].isin(['SHOT','MISS','GOAL'])) & (plays['Strength'].isin(['5x4','5x3','4x3']))).astype(np.int16)
    plays['xG_PP'] = np.nan
    plays.loc[plays['Strength'].isin(['5x4','5x3','4x3']), 'xG_PP'] = plays.loc[plays['Strength'].isin(['5x4','5x3','4x3'])]['xG']
    plays['xG_flurry_PP'] = np.nan
    plays.loc[plays['Strength'].isin(['5x4','5x3','4x3']), 'xG_flurry_PP'] = plays.loc[plays['Strength'].isin(['5x4','5x3','4x3'])]['xG_flurry']

    plays['Goals_PK'] = ((plays['Event'] == 'GOAL') & (plays['Strength'].isin(['4x5','3x5','3x4']))).astype(np.int16)
    plays['Shots_PK'] = ((plays['Event'].isin(['SHOT','GOAL'])) & (plays['Strength'].isin(['4x5','3x5','3x4']))).astype(np.int16)
    plays['ShotAttempts_PK'] = ((plays['Event'].isin(['SHOT','MISS','GOAL','BLOCK'])) & (plays['Strength'].isin(['4x5','3x5','3x4']))).astype(np.int16)
    plays['UnblockedShotAttempts_PK'] = ((plays['Event'].isin(['SHOT','MISS','GOAL','BLOCK'])) & (plays['Strength'].isin(['4x5','3x5','3x4']))).astype(np.int16)
    plays['xG_PK'] = np.nan
    plays.loc[plays['Strength'].isin(['4x5','3x5','3x4']), 'xG_PK'] = plays.loc[plays['Strength'].isin(['4x5','3x5','3x4'])]['xG']
    plays['xG_flurry_PK'] = np.nan
    plays.loc[plays['Strength'].isin(['4x5','3x5','3x4']), 'xG_flurry_PK'] = plays.loc[plays['Strength'].isin(['4x5','3x5','3x4'])]['xG_flurry']

    # adjust for score and homeaway
    plays['strength'] = plays['Strength']
    plays.loc[plays['strength'].isin(['5x4','5x3','4x3']), 'strength'] = 'PP'
    plays.loc[plays['Empty_Net'], 'strength'] = 'EN'
    plays['period'] = plays['Period']
    plays.loc[plays['period']>4, 'period'] = 4
    plays.loc[plays['period']==2, 'period'] = 1
    plays.loc[(plays['period']!=3)&(plays['Strength']!='5x5'), 'period'] = 1
    plays['scoreDiff'] = plays['Home_Score'] - plays['Away_Score']
    plays.loc[plays['scoreDiff']>3, 'scoreDiff'] = 3
    plays.loc[plays['scoreDiff']<-3, 'scoreDiff'] = -3
    plays.loc[(plays['scoreDiff']>1)&(plays['Strength']!='5x5'), 'scoreDiff'] = 1
    plays.loc[(plays['scoreDiff']<-1)&(plays['Strength']!='5x5'), 'scoreDiff'] = -1
    plays.loc[(plays['strength'].isin(['4x4','3x3']))&(plays['scoreDiff']==0), 'period'] = 1
    plays['homeAway'] = 'home'
    plays.loc[plays['Ev_Team']==plays['Away_Team'], 'homeAway'] = 'away'
    plays = plays.merge(adjustments, how='left', on=['strength', 'homeAway', 'scoreDiff', 'period'])
    plays['GoalsAdjusted'] = plays['Goals'] * plays['goalsAdjustment'].fillna(1.0)
    plays.loc[(plays['Away_Goalie'].isnull())&(plays['homeAway']=='home'), 'GoalsAdjusted'] = 0
    plays.loc[(plays['Home_Goalie'].isnull())&(plays['homeAway']=='away'), 'GoalsAdjusted'] = 0
    plays['ShotsAdjusted'] = plays['Shots'] * plays['shotsAdjustment'].fillna(1.0)
    plays['ShotAttemptsAdjusted'] = plays['ShotAttempts'] * plays['shotAttemptsAdjustment'].fillna(1.0)
    plays['UnblockedShotAttemptsAdjusted'] = plays['UnblockedShotAttempts'] * plays['unblockedShotAttemptsAdjustment'].fillna(1.0)
    plays['xGAdjusted'] = plays['xG'] * plays['xGAdjustment'].fillna(1.0)
    plays['xG_flurryAdjusted'] = plays['xG_flurry'] * plays['xGAdjustment'].fillna(1.0)
    plays['GoalsAdjusted_5v5'] = plays['Goals_5v5'] * plays['goalsAdjustment'].fillna(1.0)
    plays.loc[(plays['Away_Goalie'].isnull())&(plays['homeAway']=='home'), 'GoalsAdjusted_5v5'] = 0
    plays.loc[(plays['Home_Goalie'].isnull())&(plays['homeAway']=='away'), 'GoalsAdjusted_5v5'] = 0
    plays['ShotsAdjusted_5v5'] = plays['Shots_5v5'] * plays['shotsAdjustment'].fillna(1.0)
    plays['ShotAttemptsAdjusted_5v5'] = plays['ShotAttempts_5v5'] * plays['shotAttemptsAdjustment'].fillna(1.0)
    plays['UnblockedShotAttemptsAdjusted_5v5'] = plays['UnblockedShotAttempts_5v5'] * plays['unblockedShotAttemptsAdjustment'].fillna(1.0)
    plays['xGAdjusted_5v5'] = plays['xG_5v5'] * plays['xGAdjustment'].fillna(1.0)
    plays['xG_flurryAdjusted_5v5'] = plays['xG_flurry_5v5'] * plays['xGAdjustment'].fillna(1.0)
    plays['GoalsAdjusted_PP'] = plays['Goals_PP'] * plays['goalsAdjustment'].fillna(1.0)
    plays.loc[(plays['Away_Goalie'].isnull())&(plays['homeAway']=='home'), 'GoalsAdjusted_PP'] = 0
    plays.loc[(plays['Home_Goalie'].isnull())&(plays['homeAway']=='away'), 'GoalsAdjusted_PP'] = 0
    plays['ShotsAdjusted_PP'] = plays['Shots_PP'] * plays['shotsAdjustment'].fillna(1.0)
    plays['ShotAttemptsAdjusted_PP'] = plays['ShotAttempts_PP'] * plays['shotAttemptsAdjustment'].fillna(1.0)
    plays['UnblockedShotAttemptsAdjusted_PP'] = plays['UnblockedShotAttempts_PP'] * plays['unblockedShotAttemptsAdjustment'].fillna(1.0)
    plays['xGAdjusted_PP'] = plays['xG_PP'] * plays['xGAdjustment'].fillna(1.0)
    plays['xG_flurryAdjusted_PP'] = plays['xG_flurry_PP'] * plays['xGAdjustment'].fillna(1.0)

    # create field for player that should get credited with the event
    plays['Player'] = np.nan
    plays['PlayerID'] = np.nan
    plays.loc[plays['Event']=='BLOCK', 'Player'] = plays.loc[plays['Event']=='BLOCK']['p2_name']
    plays.loc[plays['Event']=='BLOCK', 'PlayerID'] = plays.loc[plays['Event']=='BLOCK']['p2_ID']
    plays.loc[plays['Event'].isin(['SHOT','MISS','GOAL','PENL']), 'Player'] = plays.loc[plays['Event'].isin(['SHOT','MISS','GOAL','PENL'])]['p1_name']
    plays.loc[plays['Event'].isin(['SHOT','MISS','GOAL','PENL']), 'PlayerID'] = plays.loc[plays['Event'].isin(['SHOT','MISS','GOAL','PENL'])]['p1_ID']

    # create first aggregate
    playerGame = plays.groupby([
            'Game_Id','Date','Player','PlayerID','Season'
        ]).agg({
            'Goals' : sum,
            'Shootout_Goals' : sum,
            'xG' : sum,
            'xG_flurry' : sum,
            'Shots' : sum,
            'ShotAttempts' : sum,
            'UnblockedShotAttempts' : sum,
            'Goals_5v5' : sum,
            'Shots_5v5' : sum,
            'ShotAttempts_5v5' : sum,
            'UnblockedShotAttempts_5v5' : sum,
            'xG_5v5' : sum,
            'xG_flurry_5v5' : sum,
            'Goals_PP' : sum,
            'Shots_PP' : sum,
            'ShotAttempts_PP' : sum,
            'UnblockedShotAttempts_PP' : sum,
            'xG_PP' : sum,
            'xG_flurry_PP' : sum,
            'Goals_PK' : sum,
            'Shots_PK' : sum,
            'ShotAttempts_PK' : sum,
            'UnblockedShotAttempts_PK' : sum,
            'xG_PK' : sum,
            'xG_flurry_PK' : sum,
            'GoalsAdjusted' : sum,
            'xGAdjusted' : sum,
            'xG_flurryAdjusted' : sum,
            'ShotsAdjusted' : sum,
            'ShotAttemptsAdjusted' : sum,
            'UnblockedShotAttemptsAdjusted' : sum,
            'GoalsAdjusted_5v5' : sum,
            'ShotsAdjusted_5v5' : sum,
            'ShotAttemptsAdjusted_5v5' : sum,
            'UnblockedShotAttemptsAdjusted_5v5' : sum,
            'xGAdjusted_5v5' : sum,
            'xG_flurryAdjusted_5v5' : sum,
            'GoalsAdjusted_PP' : sum,
            'ShotsAdjusted_PP' : sum,
            'ShotAttemptsAdjusted_PP' : sum,
            'UnblockedShotAttemptsAdjusted_PP' : sum,
            'xGAdjusted_PP' : sum,
            'xG_flurryAdjusted_PP' : sum,
            'Penalties' : sum
    }).reset_index()

    playerGame['Goals'] = playerGame['Goals'].astype(np.int16)
    playerGame['Shootout_Goals'] = playerGame['Shootout_Goals'].astype(np.int16)
    playerGame['Shots'] = playerGame['Shots'].astype(np.int32)
    playerGame['ShotAttempts'] = playerGame['ShotAttempts'].astype(np.int32)
    playerGame['Goals_5v5'] = playerGame['Goals_5v5'].astype(np.int16)
    playerGame['Shots_5v5'] = playerGame['Shots_5v5'].astype(np.int32)
    playerGame['ShotAttempts_5v5'] = playerGame['ShotAttempts_5v5'].astype(np.int32)
    playerGame['Goals_PP'] = playerGame['Goals_PP'].astype(np.int16)
    playerGame['Shots_PP'] = playerGame['Shots_PP'].astype(np.int32)
    playerGame['ShotAttempts_PP'] = playerGame['ShotAttempts_PP'].astype(np.int32)
    playerGame['Goals_PK'] = playerGame['Goals_PK'].astype(np.int16)
    playerGame['Shots_PK'] = playerGame['Shots_PK'].astype(np.int32)
    playerGame['ShotAttempts_PK'] = playerGame['ShotAttempts_PK'].astype(np.int32)
    playerGame['Penalties'] = playerGame['Penalties'].astype(np.int16)

    # second aggregate for primary assists and penalties drawn
    plays = plays.rename(columns={'Goals' : 'PrimaryAssists', 'Goals_5v5' : 'PrimaryAssists_5v5',
        'Goals_PP' : 'PrimaryAssists_PP', 'Goals_PK' : 'PrimaryAssists_PK',
        'GoalsAdjusted' : 'PrimaryAssistsAdjusted', 'GoalsAdjusted_5v5' : 'PrimaryAssistsAdjusted_5v5',
        'GoalsAdjusted_PP' : 'PrimaryAssistsAdjusted_PP',
        'Penalties' : 'PenaltiesDrawn'})
    plays.loc[:,'Player'] = plays['p2_name']
    plays.loc[:,'PlayerID'] = plays['p2_ID']
    playerGame_2 = plays.groupby([
            'Game_Id','Date','Player','PlayerID','Season'
        ]).agg({
            'PrimaryAssists' : sum,
            'PrimaryAssists_5v5' : sum,
            'PrimaryAssists_PP' : sum,
            'PrimaryAssists_PK' : sum,
            'PrimaryAssistsAdjusted' : sum,
            'PrimaryAssistsAdjusted_5v5' : sum,
            'PrimaryAssistsAdjusted_PP' : sum,
            'PenaltiesDrawn' : sum
    }).reset_index()
    playerGame = playerGame.merge(playerGame_2, how='left', on=['Game_Id','Date','Player','PlayerID','Season'])
    del playerGame_2

    playerGame['PrimaryAssists'] = playerGame['PrimaryAssists'].fillna(0).astype(np.int16)
    playerGame['PrimaryAssists_5v5'] = playerGame['PrimaryAssists_5v5'].fillna(0).astype(np.int16)
    playerGame['PrimaryAssists_PP'] = playerGame['PrimaryAssists_PP'].fillna(0).astype(np.int16)
    playerGame['PrimaryAssists_PK'] = playerGame['PrimaryAssists_PK'].fillna(0).astype(np.int16)
    playerGame['PenaltiesDrawn'] = playerGame['PenaltiesDrawn'].fillna(0).astype(np.int16)

    # third aggregate for secondary assists
    plays = plays.rename(columns={'PrimaryAssists' : 'SecondaryAssists', 'PrimaryAssists_5v5' : 'SecondaryAssists_5v5',
        'PrimaryAssists_PP' : 'SecondaryAssists_PP', 'PrimaryAssists_PK' : 'SecondaryAssists_PK',
        'PrimaryAssistsAdjusted' : 'SecondaryAssistsAdjusted', 'PrimaryAssistsAdjusted_5v5' : 'SecondaryAssistsAdjusted_5v5',
        'PrimaryAssistsAdjusted_PP' : 'SecondaryAssistsAdjusted_PP'})
    plays.loc[:,'Player'] = plays['p3_name']
    plays.loc[:,'PlayerID'] = plays['p3_ID']
    playerGame_3 = plays.groupby([
            'Game_Id','Date','Player','PlayerID','Season'
        ]).agg({
            'SecondaryAssists' : sum,
            'SecondaryAssists_5v5' : sum,
            'SecondaryAssists_PP' : sum,
            'SecondaryAssists_PK' : sum,
            'SecondaryAssistsAdjusted' : sum,
            'SecondaryAssistsAdjusted_5v5' : sum,
            'SecondaryAssistsAdjusted_PP' : sum
    }).reset_index()
    playerGame = playerGame.merge(playerGame_3, how='left', on=['Game_Id','Date','Player','PlayerID','Season'])
    del playerGame_3

    playerGame['SecondaryAssists'] = playerGame['SecondaryAssists'].fillna(0).astype(np.int16)
    playerGame['SecondaryAssists_5v5'] = playerGame['SecondaryAssists_5v5'].fillna(0).astype(np.int16)
    playerGame['SecondaryAssists_PP'] = playerGame['SecondaryAssists_PP'].fillna(0).astype(np.int16)
    playerGame['SecondaryAssists_PK'] = playerGame['SecondaryAssists_PK'].fillna(0).astype(np.int16)

    # prep data for on-ice aggregates
    shots = plays.loc[plays['Event'].isin(['SHOT','MISS','GOAL','BLOCK','FAC'])]
    shots = shots.rename(columns={'SecondaryAssists' : 'Goals', 'SecondaryAssists_5v5' : 'Goals_5v5',
        'SecondaryAssists_PP' : 'Goals_PP', 'SecondaryAssists_PK' : 'Goals_PK',
        'SecondaryAssistsAdjusted' : 'GoalsAdjusted', 'SecondaryAssistsAdjusted_5v5' : 'GoalsAdjusted_5v5',
        'SecondaryAssistsAdjusted_PP' : 'GoalsAdjusted_PP', 'PenaltiesDrawn' : 'Penalties'})
    del plays # to save some memory
    shots['HomeTeamEvent'] = ((shots['Ev_Team']==shots['Home_Team'])|((shots['Ev_Team']==shots['Away_Team'])&(shots['Event']=='BLOCK'))).astype(int)
    shots['AwayTeamEvent'] = ((shots['Ev_Team']==shots['Away_Team'])|((shots['Ev_Team']==shots['Home_Team'])&(shots['Event']=='BLOCK'))).astype(int)

    # on-ice stats home players
    for i in range(1,7):
        shots['Player'] = shots['homePlayer{}'.format(str(i))].copy()
        shots['PlayerID'] = shots['homePlayer{}_id'.format(str(i))].copy()

        shots['Goals_onice_home{}'.format(str(i))] = shots['Goals'] * shots['HomeTeamEvent']
        shots['Shots_onice_home{}'.format(str(i))] = shots['Shots'] * shots['HomeTeamEvent']
        shots['ShotAttempts_onice_home{}'.format(str(i))] = shots['ShotAttempts'] * shots['HomeTeamEvent']
        shots['UnblockedShotAttempts_onice_home{}'.format(str(i))] = shots['UnblockedShotAttempts'] * shots['HomeTeamEvent']
        shots['xG_onice_home{}'.format(str(i))] = shots['xG'] * shots['HomeTeamEvent']
        shots['xG_flurry_onice_home{}'.format(str(i))] = shots['xG_flurry'] * shots['HomeTeamEvent']
        shots['Goals_5v5_onice_home{}'.format(str(i))] = shots['Goals_5v5'] * shots['HomeTeamEvent']
        shots['Shots_5v5_onice_home{}'.format(str(i))] = shots['Shots_5v5'] * shots['HomeTeamEvent']
        shots['ShotAttempts_5v5_onice_home{}'.format(str(i))] = shots['ShotAttempts_5v5'] * shots['HomeTeamEvent']
        shots['UnblockedShotAttempts_5v5_onice_home{}'.format(str(i))] = shots['UnblockedShotAttempts_5v5'] * shots['HomeTeamEvent']
        shots['xG_5v5_onice_home{}'.format(str(i))] = shots['xG_5v5'] * shots['HomeTeamEvent']
        shots['xG_flurry_5v5_onice_home{}'.format(str(i))] = shots['xG_flurry_5v5'] * shots['HomeTeamEvent']
        shots['Goals_PP_onice_home{}'.format(str(i))] = shots['Goals_PP'] * shots['HomeTeamEvent']
        shots['Shots_PP_onice_home{}'.format(str(i))] = shots['Shots_PP'] * shots['HomeTeamEvent']
        shots['ShotAttempts_PP_onice_home{}'.format(str(i))] = shots['ShotAttempts_PP'] * shots['HomeTeamEvent']
        shots['UnblockedShotAttempts_PP_onice_home{}'.format(str(i))] = shots['UnblockedShotAttempts_PP'] * shots['HomeTeamEvent']
        shots['xG_PP_onice_home{}'.format(str(i))] = shots['xG_PP'] * shots['HomeTeamEvent']
        shots['xG_flurry_PP_onice_home{}'.format(str(i))] = shots['xG_flurry_PP'] * shots['HomeTeamEvent']
        shots['Goals_PK_onice_home{}'.format(str(i))] = shots['Goals_PK'] * shots['HomeTeamEvent']
        shots['Shots_PK_onice_home{}'.format(str(i))] = shots['Shots_PK'] * shots['HomeTeamEvent']
        shots['ShotAttempts_PK_onice_home{}'.format(str(i))] = shots['ShotAttempts_PK'] * shots['HomeTeamEvent']
        shots['UnblockedShotAttempts_PK_onice_home{}'.format(str(i))] = shots['UnblockedShotAttempts_PK'] * shots['HomeTeamEvent']
        shots['xG_PK_onice_home{}'.format(str(i))] = shots['xG_PK'] * shots['HomeTeamEvent']
        shots['xG_flurry_PK_onice_home{}'.format(str(i))] = shots['xG_flurry_PK'] * shots['HomeTeamEvent']

        shots['GoalsAdjusted_onice_home{}'.format(str(i))] = shots['GoalsAdjusted'] * shots['HomeTeamEvent']
        shots['ShotsAdjusted_onice_home{}'.format(str(i))] = shots['ShotsAdjusted'] * shots['HomeTeamEvent']
        shots['ShotAttemptsAdjusted_onice_home{}'.format(str(i))] = shots['ShotAttemptsAdjusted'] * shots['HomeTeamEvent']
        shots['UnblockedShotAttemptsAdjusted_onice_home{}'.format(str(i))] = shots['UnblockedShotAttemptsAdjusted'] * shots['HomeTeamEvent']
        shots['xGAdjusted_onice_home{}'.format(str(i))] = shots['xGAdjusted'] * shots['HomeTeamEvent']
        shots['xG_flurryAdjusted_onice_home{}'.format(str(i))] = shots['xG_flurryAdjusted'] * shots['HomeTeamEvent']
        shots['GoalsAdjusted_5v5_onice_home{}'.format(str(i))] = shots['GoalsAdjusted_5v5'] * shots['HomeTeamEvent']
        shots['ShotsAdjusted_5v5_onice_home{}'.format(str(i))] = shots['ShotsAdjusted_5v5'] * shots['HomeTeamEvent']
        shots['ShotAttemptsAdjusted_5v5_onice_home{}'.format(str(i))] = shots['ShotAttemptsAdjusted_5v5'] * shots['HomeTeamEvent']
        shots['UnblockedShotAttemptsAdjusted_5v5_onice_home{}'.format(str(i))] = shots['UnblockedShotAttemptsAdjusted_5v5'] * shots['HomeTeamEvent']
        shots['xGAdjusted_5v5_onice_home{}'.format(str(i))] = shots['xGAdjusted_5v5'] * shots['HomeTeamEvent']
        shots['xG_flurryAdjusted_5v5_onice_home{}'.format(str(i))] = shots['xG_flurryAdjusted_5v5'] * shots['HomeTeamEvent']
        shots['GoalsAdjusted_PP_onice_home{}'.format(str(i))] = shots['GoalsAdjusted_PP'] * shots['HomeTeamEvent']
        shots['ShotsAdjusted_PP_onice_home{}'.format(str(i))] = shots['ShotsAdjusted_PP'] * shots['HomeTeamEvent']
        shots['ShotAttemptsAdjusted_PP_onice_home{}'.format(str(i))] = shots['ShotAttemptsAdjusted_PP'] * shots['HomeTeamEvent']
        shots['UnblockedShotAttemptsAdjusted_PP_onice_home{}'.format(str(i))] = shots['UnblockedShotAttemptsAdjusted_PP'] * shots['HomeTeamEvent']
        shots['xGAdjusted_PP_onice_home{}'.format(str(i))] = shots['xGAdjusted_PP'] * shots['HomeTeamEvent']
        shots['xG_flurryAdjusted_PP_onice_home{}'.format(str(i))] = shots['xG_flurryAdjusted_PP'] * shots['HomeTeamEvent']

        shots['GoalsAgainst_onice_home{}'.format(str(i))] = shots['Goals'] * shots['AwayTeamEvent']
        shots['ShotsAgainst_onice_home{}'.format(str(i))] = shots['Shots'] * shots['AwayTeamEvent']
        shots['ShotAttemptsAgainst_onice_home{}'.format(str(i))] = shots['ShotAttempts'] * shots['AwayTeamEvent']
        shots['UnblockedShotAttemptsAgainst_onice_home{}'.format(str(i))] = shots['UnblockedShotAttempts'] * shots['AwayTeamEvent']
        shots['xGAgainst_onice_home{}'.format(str(i))] = shots['xG'] * shots['AwayTeamEvent']
        shots['xG_flurryAgainst_onice_home{}'.format(str(i))] = shots['xG_flurry'] * shots['AwayTeamEvent']
        shots['GoalsAgainst_5v5_onice_home{}'.format(str(i))] = shots['Goals_5v5'] * shots['AwayTeamEvent']
        shots['ShotsAgainst_5v5_onice_home{}'.format(str(i))] = shots['Shots_5v5'] * shots['AwayTeamEvent']
        shots['ShotAttemptsAgainst_5v5_onice_home{}'.format(str(i))] = shots['ShotAttempts_5v5'] * shots['AwayTeamEvent']
        shots['UnblockedShotAttemptsAgainst_5v5_onice_home{}'.format(str(i))] = shots['UnblockedShotAttempts_5v5'] * shots['AwayTeamEvent']
        shots['xGAgainst_5v5_onice_home{}'.format(str(i))] = shots['xG_5v5'] * shots['AwayTeamEvent']
        shots['xG_flurryAgainst_5v5_onice_home{}'.format(str(i))] = shots['xG_flurry_5v5'] * shots['AwayTeamEvent']
        shots['GoalsAgainst_PP_onice_home{}'.format(str(i))] = shots['Goals_PK'] * shots['AwayTeamEvent']
        shots['ShotsAgainst_PP_onice_home{}'.format(str(i))] = shots['Shots_PK'] * shots['AwayTeamEvent']
        shots['ShotAttemptsAgainst_PP_onice_home{}'.format(str(i))] = shots['ShotAttempts_PK'] * shots['AwayTeamEvent']
        shots['UnblockedShotAttemptsAgainst_PP_onice_home{}'.format(str(i))] = shots['UnblockedShotAttempts_PK'] * shots['AwayTeamEvent']
        shots['xGAgainst_PP_onice_home{}'.format(str(i))] = shots['xG_PK'] * shots['AwayTeamEvent']
        shots['xG_flurryAgainst_PP_onice_home{}'.format(str(i))] = shots['xG_flurry_PK'] * shots['AwayTeamEvent']
        shots['GoalsAgainst_PK_onice_home{}'.format(str(i))] = shots['Goals_PP'] * shots['AwayTeamEvent']
        shots['ShotsAgainst_PK_onice_home{}'.format(str(i))] = shots['Shots_PP'] * shots['AwayTeamEvent']
        shots['ShotAttemptsAgainst_PK_onice_home{}'.format(str(i))] = shots['ShotAttempts_PP'] * shots['AwayTeamEvent']
        shots['UnblockedShotAttemptsAgainst_PK_onice_home{}'.format(str(i))] = shots['UnblockedShotAttempts_PP'] * shots['AwayTeamEvent']
        shots['xGAgainst_PK_onice_home{}'.format(str(i))] = shots['xG_PP'] * shots['AwayTeamEvent']
        shots['xG_flurryAgainst_PK_onice_home{}'.format(str(i))] = shots['xG_flurry_PP'] * shots['AwayTeamEvent']
        shots['ReboundShotsAgainst_onice_home{}'.format(str(i))] = (shots['ShotCategory']=='Rebound').astype(np.int16) * shots['AwayTeamEvent']

        shots['GoalsAdjustedAgainst_onice_home{}'.format(str(i))] = shots['GoalsAdjusted'] * shots['AwayTeamEvent']
        shots['ShotsAdjustedAgainst_onice_home{}'.format(str(i))] = shots['ShotsAdjusted'] * shots['AwayTeamEvent']
        shots['ShotAttemptsAdjustedAgainst_onice_home{}'.format(str(i))] = shots['ShotAttemptsAdjusted'] * shots['AwayTeamEvent']
        shots['UnblockedShotAttemptsAdjustedAgainst_onice_home{}'.format(str(i))] = shots['UnblockedShotAttemptsAdjusted'] * shots['AwayTeamEvent']
        shots['xGAdjustedAgainst_onice_home{}'.format(str(i))] = shots['xGAdjusted'] * shots['AwayTeamEvent']
        shots['xG_flurryAdjustedAgainst_onice_home{}'.format(str(i))] = shots['xG_flurryAdjusted'] * shots['AwayTeamEvent']
        shots['GoalsAdjustedAgainst_5v5_onice_home{}'.format(str(i))] = shots['GoalsAdjusted_5v5'] * shots['AwayTeamEvent']
        shots['ShotsAdjustedAgainst_5v5_onice_home{}'.format(str(i))] = shots['ShotsAdjusted_5v5'] * shots['AwayTeamEvent']
        shots['ShotAttemptsAdjustedAgainst_5v5_onice_home{}'.format(str(i))] = shots['ShotAttemptsAdjusted_5v5'] * shots['AwayTeamEvent']
        shots['UnblockedShotAttemptsAdjustedAgainst_5v5_onice_home{}'.format(str(i))] = shots['UnblockedShotAttemptsAdjusted_5v5'] * shots['AwayTeamEvent']
        shots['xGAdjustedAgainst_5v5_onice_home{}'.format(str(i))] = shots['xGAdjusted_5v5'] * shots['AwayTeamEvent']
        shots['xG_flurryAdjustedAgainst_5v5_onice_home{}'.format(str(i))] = shots['xG_flurryAdjusted_5v5'] * shots['AwayTeamEvent']
        shots['GoalsAdjustedAgainst_PK_onice_home{}'.format(str(i))] = shots['GoalsAdjusted_PP'] * shots['AwayTeamEvent']
        shots['ShotsAdjustedAgainst_PK_onice_home{}'.format(str(i))] = shots['ShotsAdjusted_PP'] * shots['AwayTeamEvent']
        shots['ShotAttemptsAdjustedAgainst_PK_onice_home{}'.format(str(i))] = shots['ShotAttemptsAdjusted_PP'] * shots['AwayTeamEvent']
        shots['UnblockedShotAttemptsAdjustedAgainst_PK_onice_home{}'.format(str(i))] = shots['UnblockedShotAttemptsAdjusted_PP'] * shots['AwayTeamEvent']
        shots['xGAdjustedAgainst_PK_onice_home{}'.format(str(i))] = shots['xGAdjusted_PP'] * shots['AwayTeamEvent']
        shots['xG_flurryAdjustedAgainst_PK_onice_home{}'.format(str(i))] = shots['xG_flurryAdjusted_PP'] * shots['AwayTeamEvent']

        shots['team_home{}'.format(str(i))] = shots['Home_Team'].copy()

        # add column for tracking player positions
        shots['position_home{}'.format(str(i))] = 1

        playerGame_home = shots.groupby([
                'Game_Id','Date','Player','PlayerID','Season'
            ]).agg({
                'Goals_onice_home{}'.format(str(i)) : sum,
                'Shots_onice_home{}'.format(str(i)) : sum,
                'ShotAttempts_onice_home{}'.format(str(i)) : sum,
                'UnblockedShotAttempts_onice_home{}'.format(str(i)) : sum,
                'xG_onice_home{}'.format(str(i)) : sum,
                'xG_flurry_onice_home{}'.format(str(i)) : sum,
                'Goals_5v5_onice_home{}'.format(str(i)) : sum,
                'Shots_5v5_onice_home{}'.format(str(i)) : sum,
                'ShotAttempts_5v5_onice_home{}'.format(str(i)) : sum,
                'UnblockedShotAttempts_5v5_onice_home{}'.format(str(i)) : sum,
                'xG_5v5_onice_home{}'.format(str(i)) : sum,
                'xG_flurry_5v5_onice_home{}'.format(str(i)) : sum,
                'Goals_PP_onice_home{}'.format(str(i)) : sum,
                'Shots_PP_onice_home{}'.format(str(i)) : sum,
                'ShotAttempts_PP_onice_home{}'.format(str(i)) : sum,
                'UnblockedShotAttempts_PP_onice_home{}'.format(str(i)) : sum,
                'xG_PP_onice_home{}'.format(str(i)) : sum,
                'xG_flurry_PP_onice_home{}'.format(str(i)) : sum,
                'Goals_PK_onice_home{}'.format(str(i)) : sum,
                'Shots_PK_onice_home{}'.format(str(i)) : sum,
                'ShotAttempts_PK_onice_home{}'.format(str(i)) : sum,
                'UnblockedShotAttempts_PK_onice_home{}'.format(str(i)) : sum,
                'xG_PK_onice_home{}'.format(str(i)) : sum,
                'xG_flurry_PK_onice_home{}'.format(str(i)) : sum,
                'GoalsAgainst_onice_home{}'.format(str(i)) : sum,
                'ShotsAgainst_onice_home{}'.format(str(i)) : sum,
                'ShotAttemptsAgainst_onice_home{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAgainst_onice_home{}'.format(str(i)) : sum,
                'xGAgainst_onice_home{}'.format(str(i)) : sum,
                'xG_flurryAgainst_onice_home{}'.format(str(i)) : sum,
                'GoalsAgainst_5v5_onice_home{}'.format(str(i)) : sum,
                'ShotsAgainst_5v5_onice_home{}'.format(str(i)) : sum,
                'ShotAttemptsAgainst_5v5_onice_home{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAgainst_5v5_onice_home{}'.format(str(i)) : sum,
                'xGAgainst_5v5_onice_home{}'.format(str(i)) : sum,
                'xG_flurryAgainst_5v5_onice_home{}'.format(str(i)) : sum,
                'GoalsAgainst_PP_onice_home{}'.format(str(i)) : sum,
                'ShotsAgainst_PP_onice_home{}'.format(str(i)) : sum,
                'ShotAttemptsAgainst_PP_onice_home{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAgainst_PP_onice_home{}'.format(str(i)) : sum,
                'xGAgainst_PP_onice_home{}'.format(str(i)) : sum,
                'xG_flurryAgainst_PP_onice_home{}'.format(str(i)) : sum,
                'GoalsAgainst_PK_onice_home{}'.format(str(i)) : sum,
                'ShotsAgainst_PK_onice_home{}'.format(str(i)) : sum,
                'ShotAttemptsAgainst_PK_onice_home{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAgainst_PK_onice_home{}'.format(str(i)) : sum,
                'xGAgainst_PK_onice_home{}'.format(str(i)) : sum,
                'xG_flurryAgainst_PK_onice_home{}'.format(str(i)) : sum,
                'GoalsAdjusted_onice_home{}'.format(str(i)) : sum,
                'ShotsAdjusted_onice_home{}'.format(str(i)) : sum,
                'ShotAttemptsAdjusted_onice_home{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAdjusted_onice_home{}'.format(str(i)) : sum,
                'xGAdjusted_onice_home{}'.format(str(i)) : sum,
                'xG_flurryAdjusted_onice_home{}'.format(str(i)) : sum,
                'GoalsAdjusted_5v5_onice_home{}'.format(str(i)) : sum,
                'ShotsAdjusted_5v5_onice_home{}'.format(str(i)) : sum,
                'ShotAttemptsAdjusted_5v5_onice_home{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAdjusted_5v5_onice_home{}'.format(str(i)) : sum,
                'xGAdjusted_5v5_onice_home{}'.format(str(i)) : sum,
                'xG_flurryAdjusted_5v5_onice_home{}'.format(str(i)) : sum,
                'GoalsAdjusted_PP_onice_home{}'.format(str(i)) : sum,
                'ShotsAdjusted_PP_onice_home{}'.format(str(i)) : sum,
                'ShotAttemptsAdjusted_PP_onice_home{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAdjusted_PP_onice_home{}'.format(str(i)) : sum,
                'xGAdjusted_PP_onice_home{}'.format(str(i)) : sum,
                'xG_flurryAdjusted_PP_onice_home{}'.format(str(i)) : sum,
                'GoalsAdjustedAgainst_onice_home{}'.format(str(i)) : sum,
                'ShotsAdjustedAgainst_onice_home{}'.format(str(i)) : sum,
                'ShotAttemptsAdjustedAgainst_onice_home{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAdjustedAgainst_onice_home{}'.format(str(i)) : sum,
                'xGAdjustedAgainst_onice_home{}'.format(str(i)) : sum,
                'xG_flurryAdjustedAgainst_onice_home{}'.format(str(i)) : sum,
                'GoalsAdjustedAgainst_5v5_onice_home{}'.format(str(i)) : sum,
                'ShotsAdjustedAgainst_5v5_onice_home{}'.format(str(i)) : sum,
                'ShotAttemptsAdjustedAgainst_5v5_onice_home{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAdjustedAgainst_5v5_onice_home{}'.format(str(i)) : sum,
                'xGAdjustedAgainst_5v5_onice_home{}'.format(str(i)) : sum,
                'xG_flurryAdjustedAgainst_5v5_onice_home{}'.format(str(i)) : sum,
                'GoalsAdjustedAgainst_PK_onice_home{}'.format(str(i)) : sum,
                'ShotsAdjustedAgainst_PK_onice_home{}'.format(str(i)) : sum,
                'ShotAttemptsAdjustedAgainst_PK_onice_home{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAdjustedAgainst_PK_onice_home{}'.format(str(i)) : sum,
                'xGAdjustedAgainst_PK_onice_home{}'.format(str(i)) : sum,
                'xG_flurryAdjustedAgainst_PK_onice_home{}'.format(str(i)) : sum,
                'ReboundShotsAgainst_onice_home{}'.format(str(i)) : sum,
                'position_home{}'.format(str(i)) : sum,
                'team_home{}'.format(str(i)) : lambda x: x.value_counts().index[0]
        }).reset_index().fillna(0)
        playerGame_home['team_home{}'.format(str(i))] = playerGame_home['team_home{}'.format(str(i))].replace(0, '')

        playerGame = playerGame.merge(playerGame_home, how='outer', on=['Game_Id','Date','Player','PlayerID','Season'])
        del playerGame_home

    # on-ice stats away players
    for i in range(1,7):
        shots['Player'] = shots['awayPlayer{}'.format(str(i))].copy()
        shots['PlayerID'] = shots['awayPlayer{}_id'.format(str(i))].copy()

        shots['Goals_onice_away{}'.format(str(i))] = shots['Goals'] * shots['AwayTeamEvent']
        shots['Shots_onice_away{}'.format(str(i))] = shots['Shots'] * shots['AwayTeamEvent']
        shots['ShotAttempts_onice_away{}'.format(str(i))] = shots['ShotAttempts'] * shots['AwayTeamEvent']
        shots['UnblockedShotAttempts_onice_away{}'.format(str(i))] = shots['UnblockedShotAttempts'] * shots['AwayTeamEvent']
        shots['xG_onice_away{}'.format(str(i))] = shots['xG'] * shots['AwayTeamEvent']
        shots['xG_flurry_onice_away{}'.format(str(i))] = shots['xG_flurry'] * shots['AwayTeamEvent']
        shots['Goals_5v5_onice_away{}'.format(str(i))] = shots['Goals_5v5'] * shots['AwayTeamEvent']
        shots['Shots_5v5_onice_away{}'.format(str(i))] = shots['Shots_5v5'] * shots['AwayTeamEvent']
        shots['ShotAttempts_5v5_onice_away{}'.format(str(i))] = shots['ShotAttempts_5v5'] * shots['AwayTeamEvent']
        shots['UnblockedShotAttempts_5v5_onice_away{}'.format(str(i))] = shots['UnblockedShotAttempts_5v5'] * shots['AwayTeamEvent']
        shots['xG_5v5_onice_away{}'.format(str(i))] = shots['xG_5v5'] * shots['AwayTeamEvent']
        shots['xG_flurry_5v5_onice_away{}'.format(str(i))] = shots['xG_flurry_5v5'] * shots['AwayTeamEvent']
        shots['Goals_PP_onice_away{}'.format(str(i))] = shots['Goals_PP'] * shots['AwayTeamEvent']
        shots['Shots_PP_onice_away{}'.format(str(i))] = shots['Shots_PP'] * shots['AwayTeamEvent']
        shots['ShotAttempts_PP_onice_away{}'.format(str(i))] = shots['ShotAttempts_PP'] * shots['AwayTeamEvent']
        shots['UnblockedShotAttempts_PP_onice_away{}'.format(str(i))] = shots['UnblockedShotAttempts_PP'] * shots['AwayTeamEvent']
        shots['xG_PP_onice_away{}'.format(str(i))] = shots['xG_PP'] * shots['AwayTeamEvent']
        shots['xG_flurry_PP_onice_away{}'.format(str(i))] = shots['xG_flurry_PP'] * shots['AwayTeamEvent']
        shots['Goals_PK_onice_away{}'.format(str(i))] = shots['Goals_PK'] * shots['AwayTeamEvent']
        shots['Shots_PK_onice_away{}'.format(str(i))] = shots['Shots_PK'] * shots['AwayTeamEvent']
        shots['ShotAttempts_PK_onice_away{}'.format(str(i))] = shots['ShotAttempts_PK'] * shots['AwayTeamEvent']
        shots['UnblockedShotAttempts_PK_onice_away{}'.format(str(i))] = shots['UnblockedShotAttempts_PK'] * shots['AwayTeamEvent']
        shots['xG_PK_onice_away{}'.format(str(i))] = shots['xG_PK'] * shots['AwayTeamEvent']
        shots['xG_flurry_PK_onice_away{}'.format(str(i))] = shots['xG_flurry_PK'] * shots['AwayTeamEvent']

        shots['GoalsAdjusted_onice_away{}'.format(str(i))] = shots['GoalsAdjusted'] * shots['AwayTeamEvent']
        shots['ShotsAdjusted_onice_away{}'.format(str(i))] = shots['ShotsAdjusted'] * shots['AwayTeamEvent']
        shots['ShotAttemptsAdjusted_onice_away{}'.format(str(i))] = shots['ShotAttemptsAdjusted'] * shots['AwayTeamEvent']
        shots['UnblockedShotAttemptsAdjusted_onice_away{}'.format(str(i))] = shots['UnblockedShotAttemptsAdjusted'] * shots['AwayTeamEvent']
        shots['xGAdjusted_onice_away{}'.format(str(i))] = shots['xGAdjusted'] * shots['AwayTeamEvent']
        shots['xG_flurryAdjusted_onice_away{}'.format(str(i))] = shots['xG_flurryAdjusted'] * shots['AwayTeamEvent']
        shots['GoalsAdjusted_5v5_onice_away{}'.format(str(i))] = shots['GoalsAdjusted_5v5'] * shots['AwayTeamEvent']
        shots['ShotsAdjusted_5v5_onice_away{}'.format(str(i))] = shots['ShotsAdjusted_5v5'] * shots['AwayTeamEvent']
        shots['ShotAttemptsAdjusted_5v5_onice_away{}'.format(str(i))] = shots['ShotAttemptsAdjusted_5v5'] * shots['AwayTeamEvent']
        shots['UnblockedShotAttemptsAdjusted_5v5_onice_away{}'.format(str(i))] = shots['UnblockedShotAttemptsAdjusted_5v5'] * shots['AwayTeamEvent']
        shots['xGAdjusted_5v5_onice_away{}'.format(str(i))] = shots['xGAdjusted_5v5'] * shots['AwayTeamEvent']
        shots['xG_flurryAdjusted_5v5_onice_away{}'.format(str(i))] = shots['xG_flurryAdjusted_5v5'] * shots['AwayTeamEvent']
        shots['GoalsAdjusted_PP_onice_away{}'.format(str(i))] = shots['GoalsAdjusted_PP'] * shots['AwayTeamEvent']
        shots['ShotsAdjusted_PP_onice_away{}'.format(str(i))] = shots['ShotsAdjusted_PP'] * shots['AwayTeamEvent']
        shots['ShotAttemptsAdjusted_PP_onice_away{}'.format(str(i))] = shots['ShotAttemptsAdjusted_PP'] * shots['AwayTeamEvent']
        shots['UnblockedShotAttemptsAdjusted_PP_onice_away{}'.format(str(i))] = shots['UnblockedShotAttemptsAdjusted_PP'] * shots['AwayTeamEvent']
        shots['xGAdjusted_PP_onice_away{}'.format(str(i))] = shots['xGAdjusted_PP'] * shots['AwayTeamEvent']
        shots['xG_flurryAdjusted_PP_onice_away{}'.format(str(i))] = shots['xG_flurryAdjusted_PP'] * shots['AwayTeamEvent']

        shots['GoalsAgainst_onice_away{}'.format(str(i))] = shots['Goals'] * shots['HomeTeamEvent']
        shots['ShotsAgainst_onice_away{}'.format(str(i))] = shots['Shots'] * shots['HomeTeamEvent']
        shots['ShotAttemptsAgainst_onice_away{}'.format(str(i))] = shots['ShotAttempts'] * shots['HomeTeamEvent']
        shots['UnblockedShotAttemptsAgainst_onice_away{}'.format(str(i))] = shots['UnblockedShotAttempts'] * shots['HomeTeamEvent']
        shots['xGAgainst_onice_away{}'.format(str(i))] = shots['xG'] * shots['HomeTeamEvent']
        shots['xG_flurryAgainst_onice_away{}'.format(str(i))] = shots['xG_flurry'] * shots['HomeTeamEvent']
        shots['GoalsAgainst_5v5_onice_away{}'.format(str(i))] = shots['Goals_5v5'] * shots['HomeTeamEvent']
        shots['ShotsAgainst_5v5_onice_away{}'.format(str(i))] = shots['Shots_5v5'] * shots['HomeTeamEvent']
        shots['ShotAttemptsAgainst_5v5_onice_away{}'.format(str(i))] = shots['ShotAttempts_5v5'] * shots['HomeTeamEvent']
        shots['UnblockedShotAttemptsAgainst_5v5_onice_away{}'.format(str(i))] = shots['UnblockedShotAttempts_5v5'] * shots['HomeTeamEvent']
        shots['xGAgainst_5v5_onice_away{}'.format(str(i))] = shots['xG_5v5'] * shots['HomeTeamEvent']
        shots['xG_flurryAgainst_5v5_onice_away{}'.format(str(i))] = shots['xG_flurry_5v5'] * shots['HomeTeamEvent']
        shots['GoalsAgainst_PP_onice_away{}'.format(str(i))] = shots['Goals_PK'] * shots['HomeTeamEvent']
        shots['ShotsAgainst_PP_onice_away{}'.format(str(i))] = shots['Shots_PK'] * shots['HomeTeamEvent']
        shots['ShotAttemptsAgainst_PP_onice_away{}'.format(str(i))] = shots['ShotAttempts_PK'] * shots['HomeTeamEvent']
        shots['UnblockedShotAttemptsAgainst_PP_onice_away{}'.format(str(i))] = shots['UnblockedShotAttempts_PK'] * shots['HomeTeamEvent']
        shots['xGAgainst_PP_onice_away{}'.format(str(i))] = shots['xG_PK'] * shots['HomeTeamEvent']
        shots['xG_flurryAgainst_PP_onice_away{}'.format(str(i))] = shots['xG_flurry_PK'] * shots['HomeTeamEvent']
        shots['GoalsAgainst_PK_onice_away{}'.format(str(i))] = shots['Goals_PP'] * shots['HomeTeamEvent']
        shots['ShotsAgainst_PK_onice_away{}'.format(str(i))] = shots['Shots_PP'] * shots['HomeTeamEvent']
        shots['ShotAttemptsAgainst_PK_onice_away{}'.format(str(i))] = shots['ShotAttempts_PP'] * shots['HomeTeamEvent']
        shots['UnblockedShotAttemptsAgainst_PK_onice_away{}'.format(str(i))] = shots['UnblockedShotAttempts_PP'] * shots['HomeTeamEvent']
        shots['xGAgainst_PK_onice_away{}'.format(str(i))] = shots['xG_PP'] * shots['HomeTeamEvent']
        shots['xG_flurryAgainst_PK_onice_away{}'.format(str(i))] = shots['xG_flurry_PP'] * shots['HomeTeamEvent']
        shots['ReboundShotsAgainst_onice_away{}'.format(str(i))] = (shots['ShotCategory']=='Rebound').astype(np.int16) * shots['HomeTeamEvent']

        shots['GoalsAdjustedAgainst_onice_away{}'.format(str(i))] = shots['GoalsAdjusted'] * shots['HomeTeamEvent']
        shots['ShotsAdjustedAgainst_onice_away{}'.format(str(i))] = shots['ShotsAdjusted'] * shots['HomeTeamEvent']
        shots['ShotAttemptsAdjustedAgainst_onice_away{}'.format(str(i))] = shots['ShotAttemptsAdjusted'] * shots['HomeTeamEvent']
        shots['UnblockedShotAttemptsAdjustedAgainst_onice_away{}'.format(str(i))] = shots['UnblockedShotAttemptsAdjusted'] * shots['HomeTeamEvent']
        shots['xGAdjustedAgainst_onice_away{}'.format(str(i))] = shots['xGAdjusted'] * shots['HomeTeamEvent']
        shots['xG_flurryAdjustedAgainst_onice_away{}'.format(str(i))] = shots['xG_flurryAdjusted'] * shots['HomeTeamEvent']
        shots['GoalsAdjustedAgainst_5v5_onice_away{}'.format(str(i))] = shots['GoalsAdjusted_5v5'] * shots['HomeTeamEvent']
        shots['ShotsAdjustedAgainst_5v5_onice_away{}'.format(str(i))] = shots['ShotsAdjusted_5v5'] * shots['HomeTeamEvent']
        shots['ShotAttemptsAdjustedAgainst_5v5_onice_away{}'.format(str(i))] = shots['ShotAttemptsAdjusted_5v5'] * shots['HomeTeamEvent']
        shots['UnblockedShotAttemptsAdjustedAgainst_5v5_onice_away{}'.format(str(i))] = shots['UnblockedShotAttemptsAdjusted_5v5'] * shots['HomeTeamEvent']
        shots['xGAdjustedAgainst_5v5_onice_away{}'.format(str(i))] = shots['xGAdjusted_5v5'] * shots['HomeTeamEvent']
        shots['xG_flurryAdjustedAgainst_5v5_onice_away{}'.format(str(i))] = shots['xG_flurryAdjusted_5v5'] * shots['HomeTeamEvent']
        shots['GoalsAdjustedAgainst_PK_onice_away{}'.format(str(i))] = shots['GoalsAdjusted_PP'] * shots['HomeTeamEvent']
        shots['ShotsAdjustedAgainst_PK_onice_away{}'.format(str(i))] = shots['ShotsAdjusted_PP'] * shots['HomeTeamEvent']
        shots['ShotAttemptsAdjustedAgainst_PK_onice_away{}'.format(str(i))] = shots['ShotAttemptsAdjusted_PP'] * shots['HomeTeamEvent']
        shots['UnblockedShotAttemptsAdjustedAgainst_PK_onice_away{}'.format(str(i))] = shots['UnblockedShotAttemptsAdjusted_PP'] * shots['HomeTeamEvent']
        shots['xGAdjustedAgainst_PK_onice_away{}'.format(str(i))] = shots['xGAdjusted_PP'] * shots['HomeTeamEvent']
        shots['xG_flurryAdjustedAgainst_PK_onice_away{}'.format(str(i))] = shots['xG_flurryAdjusted_PP'] * shots['HomeTeamEvent']

        shots['team_away{}'.format(str(i))] = shots['Away_Team'].copy()

        # add column for tracking player positions
        shots['position_away{}'.format(str(i))] = 1

        playerGame_away = shots.groupby([
                'Game_Id','Date','Player','PlayerID','Season'
            ]).agg({
                'Goals_onice_away{}'.format(str(i)) : sum,
                'Shots_onice_away{}'.format(str(i)) : sum,
                'ShotAttempts_onice_away{}'.format(str(i)) : sum,
                'UnblockedShotAttempts_onice_away{}'.format(str(i)) : sum,
                'xG_onice_away{}'.format(str(i)) : sum,
                'xG_flurry_onice_away{}'.format(str(i)) : sum,
                'Goals_5v5_onice_away{}'.format(str(i)) : sum,
                'Shots_5v5_onice_away{}'.format(str(i)) : sum,
                'ShotAttempts_5v5_onice_away{}'.format(str(i)) : sum,
                'UnblockedShotAttempts_5v5_onice_away{}'.format(str(i)) : sum,
                'xG_5v5_onice_away{}'.format(str(i)) : sum,
                'xG_flurry_5v5_onice_away{}'.format(str(i)) : sum,
                'Goals_PP_onice_away{}'.format(str(i)) : sum,
                'Shots_PP_onice_away{}'.format(str(i)) : sum,
                'ShotAttempts_PP_onice_away{}'.format(str(i)) : sum,
                'UnblockedShotAttempts_PP_onice_away{}'.format(str(i)) : sum,
                'xG_PP_onice_away{}'.format(str(i)) : sum,
                'xG_flurry_PP_onice_away{}'.format(str(i)) : sum,
                'Goals_PK_onice_away{}'.format(str(i)) : sum,
                'Shots_PK_onice_away{}'.format(str(i)) : sum,
                'ShotAttempts_PK_onice_away{}'.format(str(i)) : sum,
                'UnblockedShotAttempts_PK_onice_away{}'.format(str(i)) : sum,
                'xG_PK_onice_away{}'.format(str(i)) : sum,
                'xG_flurry_PK_onice_away{}'.format(str(i)) : sum,
                'GoalsAgainst_onice_away{}'.format(str(i)) : sum,
                'ShotsAgainst_onice_away{}'.format(str(i)) : sum,
                'ShotAttemptsAgainst_onice_away{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAgainst_onice_away{}'.format(str(i)) : sum,
                'xGAgainst_onice_away{}'.format(str(i)) : sum,
                'xG_flurryAgainst_onice_away{}'.format(str(i)) : sum,
                'GoalsAgainst_5v5_onice_away{}'.format(str(i)) : sum,
                'ShotsAgainst_5v5_onice_away{}'.format(str(i)) : sum,
                'ShotAttemptsAgainst_5v5_onice_away{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAgainst_5v5_onice_away{}'.format(str(i)) : sum,
                'xGAgainst_5v5_onice_away{}'.format(str(i)) : sum,
                'xG_flurryAgainst_5v5_onice_away{}'.format(str(i)) : sum,
                'GoalsAgainst_PP_onice_away{}'.format(str(i)) : sum,
                'ShotsAgainst_PP_onice_away{}'.format(str(i)) : sum,
                'ShotAttemptsAgainst_PP_onice_away{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAgainst_PP_onice_away{}'.format(str(i)) : sum,
                'xGAgainst_PP_onice_away{}'.format(str(i)) : sum,
                'xG_flurryAgainst_PP_onice_away{}'.format(str(i)) : sum,
                'GoalsAgainst_PK_onice_away{}'.format(str(i)) : sum,
                'ShotsAgainst_PK_onice_away{}'.format(str(i)) : sum,
                'ShotAttemptsAgainst_PK_onice_away{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAgainst_PK_onice_away{}'.format(str(i)) : sum,
                'xGAgainst_PK_onice_away{}'.format(str(i)) : sum,
                'xG_flurryAgainst_PK_onice_away{}'.format(str(i)) : sum,
                'GoalsAdjusted_onice_away{}'.format(str(i)) : sum,
                'ShotsAdjusted_onice_away{}'.format(str(i)) : sum,
                'ShotAttemptsAdjusted_onice_away{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAdjusted_onice_away{}'.format(str(i)) : sum,
                'xGAdjusted_onice_away{}'.format(str(i)) : sum,
                'xG_flurryAdjusted_onice_away{}'.format(str(i)) : sum,
                'GoalsAdjusted_5v5_onice_away{}'.format(str(i)) : sum,
                'ShotsAdjusted_5v5_onice_away{}'.format(str(i)) : sum,
                'ShotAttemptsAdjusted_5v5_onice_away{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAdjusted_5v5_onice_away{}'.format(str(i)) : sum,
                'xGAdjusted_5v5_onice_away{}'.format(str(i)) : sum,
                'xG_flurryAdjusted_5v5_onice_away{}'.format(str(i)) : sum,
                'GoalsAdjusted_PP_onice_away{}'.format(str(i)) : sum,
                'ShotsAdjusted_PP_onice_away{}'.format(str(i)) : sum,
                'ShotAttemptsAdjusted_PP_onice_away{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAdjusted_PP_onice_away{}'.format(str(i)) : sum,
                'xGAdjusted_PP_onice_away{}'.format(str(i)) : sum,
                'xG_flurryAdjusted_PP_onice_away{}'.format(str(i)) : sum,
                'GoalsAdjustedAgainst_onice_away{}'.format(str(i)) : sum,
                'ShotsAdjustedAgainst_onice_away{}'.format(str(i)) : sum,
                'ShotAttemptsAdjustedAgainst_onice_away{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAdjustedAgainst_onice_away{}'.format(str(i)) : sum,
                'xGAdjustedAgainst_onice_away{}'.format(str(i)) : sum,
                'xG_flurryAdjustedAgainst_onice_away{}'.format(str(i)) : sum,
                'GoalsAdjustedAgainst_5v5_onice_away{}'.format(str(i)) : sum,
                'ShotsAdjustedAgainst_5v5_onice_away{}'.format(str(i)) : sum,
                'ShotAttemptsAdjustedAgainst_5v5_onice_away{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAdjustedAgainst_5v5_onice_away{}'.format(str(i)) : sum,
                'xGAdjustedAgainst_5v5_onice_away{}'.format(str(i)) : sum,
                'xG_flurryAdjustedAgainst_5v5_onice_away{}'.format(str(i)) : sum,
                'GoalsAdjustedAgainst_PK_onice_away{}'.format(str(i)) : sum,
                'ShotsAdjustedAgainst_PK_onice_away{}'.format(str(i)) : sum,
                'ShotAttemptsAdjustedAgainst_PK_onice_away{}'.format(str(i)) : sum,
                'UnblockedShotAttemptsAdjustedAgainst_PK_onice_away{}'.format(str(i)) : sum,
                'xGAdjustedAgainst_PK_onice_away{}'.format(str(i)) : sum,
                'xG_flurryAdjustedAgainst_PK_onice_away{}'.format(str(i)) : sum,
                'ReboundShotsAgainst_onice_away{}'.format(str(i)) : sum,
                'position_away{}'.format(str(i)) : sum,
                'team_away{}'.format(str(i)) : lambda x: x.value_counts().index[0]
        }).reset_index()
        playerGame_away['team_away{}'.format(str(i))] = playerGame_away['team_away{}'.format(str(i))].replace(0, '')

        playerGame = playerGame.merge(playerGame_away, how='outer', on=['Game_Id','Date','Player','PlayerID','Season'])
        del playerGame_away

        # add up home and away player position numbers
        playerGame['position_{}'.format(str(i))] = playerGame['position_home{}'.format(str(i))].fillna(0) + playerGame['position_away{}'.format(str(i))].fillna(0)
        playerGame = playerGame.drop(['position_home{}'.format(str(i)),'position_away{}'.format(str(i))], 1)

    # fill nulls, and add up on-ice columns so there is only one per stat
    playerGame = playerGame.fillna(0)
    for col in ['Goals_onice','Shots_onice','ShotAttempts_onice','UnblockedShotAttempts_onice','xG_onice',
                'xG_flurry_onice','Goals_5v5_onice','Shots_5v5_onice',
                'ShotAttempts_5v5_onice','UnblockedShotAttempts_5v5_onice','xG_5v5_onice',
                'xG_flurry_5v5_onice','Goals_PP_onice','Shots_PP_onice',
                'ShotAttempts_PP_onice','UnblockedShotAttempts_PP_onice','xG_PP_onice',
                'xG_flurry_PP_onice','Goals_PK_onice','Shots_PK_onice',
                'ShotAttempts_PK_onice','UnblockedShotAttempts_PK_onice','xG_PK_onice',
                'xG_flurry_PK_onice','GoalsAgainst_onice','ShotsAgainst_onice',
                'ShotAttemptsAgainst_onice','UnblockedShotAttemptsAgainst_onice',
                'xGAgainst_onice','xG_flurryAgainst_onice','GoalsAgainst_5v5_onice',
                'ShotsAgainst_5v5_onice','ShotAttemptsAgainst_5v5_onice','UnblockedShotAttemptsAgainst_5v5_onice',
                'xGAgainst_5v5_onice','xG_flurryAgainst_5v5_onice','GoalsAgainst_PP_onice',
                'ShotsAgainst_PP_onice','ShotAttemptsAgainst_PP_onice','UnblockedShotAttemptsAgainst_PP_onice',
                'xGAgainst_PP_onice','xG_flurryAgainst_PP_onice','GoalsAgainst_PK_onice',
                'ShotsAgainst_PK_onice','ShotAttemptsAgainst_PK_onice','UnblockedShotAttemptsAgainst_PK_onice',
                'xGAgainst_PK_onice','xG_flurryAgainst_PK_onice','GoalsAdjusted_onice',
                'ShotsAdjusted_onice','ShotAttemptsAdjusted_onice','UnblockedShotAttemptsAdjusted_onice',
                'xGAdjusted_onice','xG_flurryAdjusted_onice','GoalsAdjusted_5v5_onice',
                'ShotsAdjusted_5v5_onice','ShotAttemptsAdjusted_5v5_onice','UnblockedShotAttemptsAdjusted_5v5_onice',
                'xGAdjusted_5v5_onice','xG_flurryAdjusted_5v5_onice','GoalsAdjusted_PP_onice','ShotsAdjusted_PP_onice',
                'ShotAttemptsAdjusted_PP_onice','UnblockedShotAttemptsAdjusted_PP_onice',
                'xGAdjusted_PP_onice','xG_flurryAdjusted_PP_onice','GoalsAdjustedAgainst_onice','ShotsAdjustedAgainst_onice',
                'ShotAttemptsAdjustedAgainst_onice','UnblockedShotAttemptsAdjustedAgainst_onice',
                'xGAdjustedAgainst_onice','xG_flurryAdjustedAgainst_onice','GoalsAdjustedAgainst_5v5_onice','ShotsAdjustedAgainst_5v5_onice',
                'ShotAttemptsAdjustedAgainst_5v5_onice','UnblockedShotAttemptsAdjustedAgainst_5v5_onice',
                'xGAdjustedAgainst_5v5_onice','xG_flurryAdjustedAgainst_5v5_onice','GoalsAdjustedAgainst_PK_onice','ShotsAdjustedAgainst_PK_onice',
                'ShotAttemptsAdjustedAgainst_PK_onice','UnblockedShotAttemptsAdjustedAgainst_PK_onice',
                'xGAdjustedAgainst_PK_onice','xG_flurryAdjustedAgainst_PK_onice','ReboundShotsAgainst_onice']:
        playerGame[col] = playerGame['{}_home1'.format(col)] + playerGame['{}_home2'.format(col)] + playerGame['{}_home3'.format(col)] + playerGame['{}_home4'.format(col)] \
            + playerGame['{}_home5'.format(col)] + playerGame['{}_home6'.format(col)] + playerGame['{}_away1'.format(col)] + playerGame['{}_away2'.format(col)] \
            + playerGame['{}_away3'.format(col)] + playerGame['{}_away4'.format(col)] + playerGame['{}_away5'.format(col)] + playerGame['{}_away6'.format(col)]
        playerGame = playerGame.drop(['{}_home1'.format(col),'{}_home2'.format(col),'{}_home3'.format(col),'{}_home4'.format(col),'{}_home5'.format(col),
                                      '{}_home6'.format(col),'{}_away1'.format(col),'{}_away2'.format(col),'{}_away3'.format(col),'{}_away4'.format(col),
                                      '{}_away5'.format(col),'{}_away6'.format(col)], 1)

    # combine the team columns into one
    playerGame['Team'] = playerGame[['team_away1','team_away2','team_away3','team_away4','team_away5','team_away6',
                                     'team_home1','team_home2','team_home3','team_home4','team_home5','team_home6']].replace(0,'').max(1)
    playerGame = playerGame.drop(['team_away1','team_away2','team_away3','team_away4','team_away5','team_away6',
                                  'team_home1','team_home2','team_home3','team_home4','team_home5','team_home6'], 1)

    # downcast some fields
    playerGame['Game_Id'] = playerGame['Game_Id'].astype(np.int32)
    playerGame['PlayerID'] = playerGame['PlayerID'].astype(np.int32)
    playerGame['Season'] = playerGame['Season'].astype(np.int16)
    playerGame['PrimaryAssists'] = playerGame['PrimaryAssists'].astype(np.int16)
    playerGame['PrimaryAssists_5v5'] = playerGame['PrimaryAssists_5v5'].astype(np.int16)
    playerGame['PrimaryAssists_PP'] = playerGame['PrimaryAssists_PP'].astype(np.int16)
    playerGame['PrimaryAssists_PK'] = playerGame['PrimaryAssists_PK'].astype(np.int16)
    playerGame['PenaltiesDrawn'] = playerGame['PenaltiesDrawn'].astype(np.int32)
    playerGame['SecondaryAssists'] = playerGame['SecondaryAssists'].astype(np.int16)
    playerGame['SecondaryAssists_5v5'] = playerGame['SecondaryAssists_5v5'].astype(np.int16)
    playerGame['SecondaryAssists_PP'] = playerGame['SecondaryAssists_PP'].astype(np.int16)
    playerGame['SecondaryAssists_PK'] = playerGame['SecondaryAssists_PK'].astype(np.int16)
    playerGame['Goals_onice'] = playerGame['Goals_onice'].astype(np.int16)
    playerGame['Shots_onice'] = playerGame['Shots_onice'].astype(np.int32)
    playerGame['ShotAttempts_onice'] = playerGame['ShotAttempts_onice'].astype(np.int32)
    playerGame['Goals_5v5_onice'] = playerGame['Goals_5v5_onice'].astype(np.int16)
    playerGame['Shots_5v5_onice'] = playerGame['Shots_5v5_onice'].astype(np.int32)
    playerGame['ShotAttempts_5v5_onice'] = playerGame['ShotAttempts_5v5_onice'].astype(np.int32)
    playerGame['Goals_PP_onice'] = playerGame['Goals_PP_onice'].astype(np.int16)
    playerGame['Shots_PP_onice'] = playerGame['Shots_PP_onice'].astype(np.int32)
    playerGame['ShotAttempts_PP_onice'] = playerGame['ShotAttempts_PP_onice'].astype(np.int32)
    playerGame['Goals_PK_onice'] = playerGame['Goals_PK_onice'].astype(np.int16)
    playerGame['Shots_PK_onice'] = playerGame['Shots_PK_onice'].astype(np.int32)
    playerGame['ShotAttempts_PK_onice'] = playerGame['ShotAttempts_PK_onice'].astype(np.int32)
    playerGame['GoalsAgainst_onice'] = playerGame['GoalsAgainst_onice'].astype(np.int16)
    playerGame['ShotsAgainst_onice'] = playerGame['ShotsAgainst_onice'].astype(np.int32)
    playerGame['ShotAttemptsAgainst_onice'] = playerGame['ShotAttemptsAgainst_onice'].astype(np.int32)
    playerGame['GoalsAgainst_5v5_onice'] = playerGame['GoalsAgainst_5v5_onice'].astype(np.int16)
    playerGame['ShotsAgainst_5v5_onice'] = playerGame['ShotsAgainst_5v5_onice'].astype(np.int32)
    playerGame['ShotAttemptsAgainst_5v5_onice'] = playerGame['ShotAttemptsAgainst_5v5_onice'].astype(np.int32)
    playerGame['GoalsAgainst_PP_onice'] = playerGame['GoalsAgainst_PP_onice'].astype(np.int16)
    playerGame['ShotsAgainst_PP_onice'] = playerGame['ShotsAgainst_PP_onice'].astype(np.int32)
    playerGame['ShotAttemptsAgainst_PP_onice'] = playerGame['ShotAttemptsAgainst_PP_onice'].astype(np.int32)
    playerGame['GoalsAgainst_PK_onice'] = playerGame['GoalsAgainst_PK_onice'].astype(np.int16)
    playerGame['ShotsAgainst_PK_onice'] = playerGame['ShotsAgainst_PK_onice'].astype(np.int32)
    playerGame['ShotAttemptsAgainst_PK_onice'] = playerGame['ShotAttemptsAgainst_PK_onice'].astype(np.int32)

    # add positions
    players = playerGame[['Player','PlayerID','position_1','position_2','position_3','position_4','position_5','position_6']].groupby([
        'Player','PlayerID'
    ]).sum()
    players['maxcol'] = players.idxmax(1)
    players = players.reset_index()
    players['Position'] = np.nan
    players.loc[players['maxcol'].isin(['position_1','position_2','position_3']),'Position'] = 'F'
    players.loc[players['maxcol'].isin(['position_4','position_5']),'Position'] = 'D'
    players.loc[players['maxcol']=='position_6','Position'] = 'G'

    playerGame = playerGame.drop(['position_1','position_2','position_3','position_4','position_5','position_6'], 1)
    playerGame = playerGame.merge(players[['Player','PlayerID','Position']], on=['Player','PlayerID'], how='left')
    del players

    # merge with toi and zone starts data
    toi = toi.rename(columns={'Player_Id':'PlayerID'})
    zone_starts = zone_starts.rename(columns={'Player_Id':'PlayerID'})
    playerGame = playerGame.merge(toi, on=['Date','PlayerID','Player','Game_Id'], how='inner')
    playerGame = playerGame.merge(zone_starts, on=['Date','PlayerID','Player','Game_Id'], how='left')

    # add an ID field
    playerGame['PlayerGameID'] = playerGame['Date'] + '_' + playerGame['PlayerID'].astype(str)

    # add field for whether this is a playoff game or not
    playerGame['DateInt'] = playerGame['Date'].str.replace('-','').astype(np.int64)
    playerGame['teamGameRank'] = playerGame.groupby(['Team','Season'])['DateInt'].rank("dense")
    playerGame['Playoffs'] = (playerGame['teamGameRank']>82).astype(np.int8)
    playerGame.loc[(playerGame['Season']==2012)&(playerGame['teamGameRank']>48),'Playoffs'] = 1 #fix for the lockout-shortened season
    playerGame.loc[(playerGame['Season']==2019)&(playerGame['Date']>'2020-03-12'),'Playoffs'] = 1 #fix for the first covid-shortened season
    playerGame.loc[(playerGame['Season']==2020)&(playerGame['teamGameRank']>56),'Playoffs'] = 1 #fix for the second covid-shortened season
    playerGame = playerGame.drop('teamGameRank',1)
    playerGame = playerGame.loc[playerGame['PlayerID']>=1]

    return playerGame, toi_overlap

def add_elo(teamGame):
    teamGame = teamGame.sort_values(by=['DateInt','Game_Id'])
    teamGame['teamGameRankOverall'] = teamGame.groupby('Team')['DateInt'].rank("dense")
    gids = teamGame['Game_Id'].unique().tolist() #gids = teamGame.loc[teamGame['Elo'].isnull(), 'Game_Id'].unique().tolist()
    teamGame['Elo'] = 1500 #teamGame.loc[teamGame['Elo'].isnull(), 'Elo'] = 1500
    for gid in gids:
        game_df = teamGame.loc[teamGame['Game_Id']==gid].copy()
        elo_1 = game_df['Elo'].iloc[0]
        elo_2 = game_df['Elo'].iloc[1]
        if game_df['teamGameRank'].iloc[0]==1 and game_df['teamGameRankOverall'].iloc[0]>1:
            elo_1 = (elo_1*0.7) + (1505*0.3)
            teamGame.loc[(teamGame['Team']==game_df['Team'].iloc[0]) & (teamGame['teamGameRankOverall']==game_df['teamGameRankOverall'].iloc[0]), 'Elo'] = elo_1
        if game_df['teamGameRank'].iloc[1]==1 and game_df['teamGameRankOverall'].iloc[1]>1:
            elo_2 = (elo_2*0.7) + (1505*0.3)
            teamGame.loc[(teamGame['Team']==game_df['Team'].iloc[1]) & (teamGame['teamGameRankOverall']==game_df['teamGameRankOverall'].iloc[1]), 'Elo'] = elo_2
        p_1 = 1/(10**((elo_2-elo_1)/400)+1)
        p_2 = 1/(10**((elo_1-elo_2)/400)+1)
        win_1 = game_df['Win'].iloc[0]
        win_2 = game_df['Win'].iloc[1]
        victoryMarginMultiplier_1 = 0.6686 * np.log(np.max([np.abs(game_df['Goals'].iloc[0]-game_df['Goals'].iloc[1]),1])) + 0.8048
        victoryMarginMultiplier_2 = 0.6686 * np.log(np.max([np.abs(game_df['Goals'].iloc[1]-game_df['Goals'].iloc[0]),1])) + 0.8048
        if elo_1>elo_2 and win_1==1:
            autoAdjust_1 = 2.05/((elo_1-elo_2) * 0.001 + 2.05)
            autoAdjust_2 = 1
        elif elo_2>elo_1 and win_2==1:
            autoAdjust_2 = 2.05/((elo_1-elo_2) * 0.001 + 2.05)
            autoAdjust_1 = 1
        else:
            autoAdjust_1 = 1
            autoAdjust_2 = 1
        favMultiplier_1 = win_1 - p_1
        favMultiplier_2 = win_2 - p_2
        new_elo_1 = elo_1 + (6*victoryMarginMultiplier_1*autoAdjust_1*favMultiplier_1)
        new_elo_2 = elo_2 + (6*victoryMarginMultiplier_2*autoAdjust_2*favMultiplier_2)
        teamGame.loc[(teamGame['Team']==game_df['Team'].iloc[0]) & (teamGame['teamGameRankOverall']==game_df['teamGameRankOverall'].iloc[0]+1), 'Elo'] = new_elo_1
        teamGame.loc[(teamGame['Team']==game_df['Team'].iloc[1]) & (teamGame['teamGameRankOverall']==game_df['teamGameRankOverall'].iloc[1]+1), 'Elo'] = new_elo_2

    return teamGame

def aggregate_team_data(pbp, prev_teamGame, homeaway_adjustments='data/score_homeaway_adjustments.csv'):
    # read adjustments data
    adjustments = pd.read_csv(homeaway_adjustments)

    # calculate some stats in the disaggregated data
    pbp['Goals'] = ((pbp['Event'] == 'GOAL') & (pbp['Seconds_Elapsed']!=0)).astype(np.int16)
    pbp['Shootout_Goals'] = ((pbp['Event'] == 'GOAL') & (pbp['Seconds_Elapsed']==0)).astype(np.int16)
    pbp['Shots'] = ((pbp['Event'].isin(['SHOT','GOAL'])) & (pbp['Seconds_Elapsed']!=0)).astype(np.int16)
    pbp['ShotAttempts'] = ((pbp['Event'].isin(['SHOT','MISS','GOAL','BLOCK'])) & (pbp['Seconds_Elapsed']!=0)).astype(np.int16)
    pbp['UnblockedShotAttempts'] = ((pbp['Event'].isin(['SHOT','MISS','GOAL'])) & (pbp['Seconds_Elapsed']!=0)).astype(np.int16)
    pbp['Goals_5v5'] = ((pbp['Event'] == 'GOAL') & (pbp['Strength']=='5x5') & (~pbp['Empty_Net'])).astype(np.int16)
    pbp['Shots_5v5'] = ((pbp['Event'].isin(['SHOT','GOAL'])) & (pbp['Strength']=='5x5') & (~pbp['Empty_Net'])).astype(np.int16)
    pbp['ShotAttempts_5v5'] = ((pbp['Event'].isin(['SHOT','MISS','GOAL','BLOCK'])) & (pbp['Strength']=='5x5') & (~pbp['Empty_Net'])).astype(np.int16)
    pbp['UnblockedShotAttempts_5v5'] = ((pbp['Event'].isin(['SHOT','MISS','GOAL'])) & (pbp['Strength'].isin(['5x5'])) & (~pbp['Empty_Net'])).astype(np.int16)
    pbp['xG_5v5'] = np.nan
    pbp.loc[(pbp['Strength']=='5x5') & (~pbp['Empty_Net']), 'xG_5v5'] = pbp.loc[(pbp['Strength']=='5x5') & (~pbp['Empty_Net'])]['xG']
    pbp['xG_flurry_5v5'] = np.nan
    pbp.loc[pbp['Strength'].isin(['5x5']), 'xG_flurry_5v5'] = pbp.loc[(pbp['Strength'].isin(['5x5'])) & (~pbp['Empty_Net'])]['xG_flurry']
    pbp['Penalties'] = ((pbp['Event']=='PENL')&(~(pbp['Type'].str.contains('Fight')).fillna(False))).astype(np.int16)

    # add adjusted metric
    pbp['strength'] = pbp['Strength']
    pbp.loc[pbp['Ev_Team']==pbp['Away_Team'], 'strength'] = pbp.loc[pbp['Ev_Team']==pbp['Away_Team'], 'strength'].str[::-1]
    pbp.loc[pbp['strength'].isin(['5x4','5x3','4x3']), 'strength'] = 'PP'
    pbp.loc[pbp['Empty_Net'], 'strength'] = 'EN'
    pbp['period'] = pbp['Period']
    pbp.loc[pbp['period']>4, 'period'] = 4
    pbp.loc[pbp['period']==2, 'period'] = 1
    pbp.loc[(pbp['period']!=3)&(pbp['Strength']!='5x5'), 'period'] = 1
    pbp['scoreDiff'] = pbp['Home_Score'] - pbp['Away_Score']
    pbp.loc[pbp['scoreDiff']>3, 'scoreDiff'] = 3
    pbp.loc[pbp['scoreDiff']<-3, 'scoreDiff'] = -3
    pbp.loc[(pbp['scoreDiff']>1)&(pbp['Strength']!='5x5'), 'scoreDiff'] = 1
    pbp.loc[(pbp['scoreDiff']<-1)&(pbp['Strength']!='5x5'), 'scoreDiff'] = -1
    pbp.loc[(pbp['strength'].isin(['4x4','3x3']))&(pbp['scoreDiff']==0), 'period'] = 1
    pbp['homeAway'] = 'home'
    pbp.loc[pbp['Ev_Team']==pbp['Away_Team'], 'homeAway'] = 'away'
    pbp['Home'] = (pbp['homeAway']=='home').astype(np.int16)
    pbp = pbp.merge(adjustments, how='left', on=['strength', 'homeAway', 'scoreDiff', 'period'])
    pbp['GoalsAdjusted'] = pbp['Goals'] * pbp['goalsAdjustment'].fillna(1.0)
    pbp.loc[(pbp['Away_Goalie'].isnull())&(pbp['homeAway']=='home'), 'GoalsAdjusted'] = 0
    pbp.loc[(pbp['Home_Goalie'].isnull())&(pbp['homeAway']=='away'), 'GoalsAdjusted'] = 0
    pbp['ShotsAdjusted'] = pbp['Shots'] * pbp['shotsAdjustment'].fillna(1.0)
    pbp['ShotAttemptsAdjusted'] = pbp['ShotAttempts'] * pbp['shotAttemptsAdjustment'].fillna(1.0)
    pbp['UnblockedShotAttemptsAdjusted'] = pbp['UnblockedShotAttempts'] * pbp['unblockedShotAttemptsAdjustment'].fillna(1.0)
    pbp['xGAdjusted'] = pbp['xG'] * pbp['xGAdjustment'].fillna(1.0)
    pbp['xG_flurryAdjusted'] = pbp['xG_flurry'] * pbp['xGAdjustment'].fillna(1.0)
    pbp['GoalsAdjusted_5v5'] = pbp['Goals_5v5'] * pbp['goalsAdjustment'].fillna(1.0)
    pbp.loc[(pbp['Away_Goalie'].isnull())&(pbp['homeAway']=='home'), 'GoalsAdjusted_5v5'] = 0
    pbp.loc[(pbp['Home_Goalie'].isnull())&(pbp['homeAway']=='away'), 'GoalsAdjusted_5v5'] = 0
    pbp['ShotsAdjusted_5v5'] = pbp['Shots_5v5'] * pbp['shotsAdjustment'].fillna(1.0)
    pbp['ShotAttemptsAdjusted_5v5'] = pbp['ShotAttempts_5v5'] * pbp['shotAttemptsAdjustment'].fillna(1.0)
    pbp['UnblockedShotAttemptsAdjusted_5v5'] = pbp['UnblockedShotAttempts_5v5'] * pbp['unblockedShotAttemptsAdjustment'].fillna(1.0)
    pbp['xGAdjusted_5v5'] = pbp['xG_5v5'] * pbp['xGAdjustment'].fillna(1.0)
    pbp['xG_flurryAdjusted_5v5'] = pbp['xG_flurry_5v5'] * pbp['xGAdjustment'].fillna(1.0)

    # create aggregate
    teamGame = pbp.groupby([
            'Game_Id','Date','Ev_Team','Season'
        ]).agg({
            'Goals' : 'sum',
            'Shootout_Goals' : 'sum',
            'Shots' : 'sum',
            'ShotAttempts' : 'sum',
            'UnblockedShotAttempts' : 'sum',
            'xG' : 'sum',
            'xG_flurry' : 'sum',
            'Goals_5v5' : 'sum',
            'Shots_5v5' : 'sum',
            'ShotAttempts_5v5' : 'sum',
            'UnblockedShotAttempts_5v5' : 'sum',
            'xG_5v5' : 'sum',
            'xG_flurry_5v5' : 'sum',
            'GoalsAdjusted' : 'sum',
            'ShotsAdjusted' : 'sum',
            'ShotAttemptsAdjusted' : 'sum',
            'UnblockedShotAttemptsAdjusted' : 'sum',
            'xGAdjusted' : 'sum',
            'xG_flurryAdjusted' : 'sum',
            'GoalsAdjusted_5v5' : 'sum',
            'ShotsAdjusted_5v5' : 'sum',
            'ShotAttemptsAdjusted_5v5' : 'sum',
            'UnblockedShotAttemptsAdjusted_5v5' : 'sum',
            'xGAdjusted_5v5' : 'sum',
            'xG_flurryAdjusted_5v5' : 'sum',
            'Penalties' : 'sum',
            'Home' : 'max'
        }).reset_index()
    teamGame = teamGame.rename(columns={'Ev_Team':'Team'})
    teamGame = teamGame.fillna(0)

    # add starting goalies
    starting_goalies = pbp.loc[(pbp['Period']==1)&(pbp['Seconds_Elapsed']==0)]
    home_goalies = starting_goalies[['Game_Id','Date','Home_Team','Home_Goalie','Home_Goalie_Id']]
    home_goalies.columns = ['Game_Id','Date','Team','StartingGoalie','StartingGoalie_Id']
    home_goalies = home_goalies.drop_duplicates()
    away_goalies = starting_goalies[['Game_Id','Date','Away_Team','Away_Goalie','Away_Goalie_Id']]
    away_goalies.columns = ['Game_Id','Date','Team','StartingGoalie','StartingGoalie_Id']
    away_goalies = away_goalies.drop_duplicates()
    starting_goalies = pd.concat([home_goalies, away_goalies], ignore_index=True)
    starting_goalies = starting_goalies.dropna()
    teamGame = teamGame.merge(starting_goalies, on=['Game_Id','Date','Team'])

    # add win column
    df_temp = teamGame.groupby(['Game_Id','Date']).agg({'Goals':'max', 'Shootout_Goals':'max'}).reset_index()
    df_temp = df_temp.rename(columns={'Goals':'winGoals', 'Shootout_Goals':'winShootoutGoals'})
    teamGame = teamGame.merge(df_temp, on=['Game_Id','Date'])
    del df_temp
    teamGame['Win'] = ((teamGame['Goals']==teamGame['winGoals']) & (teamGame['Shootout_Goals']==teamGame['winShootoutGoals'])).astype(np.int16)
    teamGame = teamGame.drop(['Shootout_Goals','winGoals','winShootoutGoals'],1)

    # fix a bug
    teamGame = teamGame.loc[teamGame['Team'].str.len()==3]

    # add previous teamGame data, if it exists
    if prev_teamGame is not None:
        teamGame = pd.concat([prev_teamGame, teamGame], ignore_index=True)

    # sort
    teamGame = teamGame.sort_values(by=['Date','Game_Id'])

    # add field for whether this is a playoff game or not
    teamGame['DateInt'] = teamGame['Date'].str.replace('-','').astype(np.int64)
    teamGame['teamGameRank'] = teamGame.groupby(['Team','Season'])['DateInt'].rank("dense")
    teamGame['Playoffs'] = (teamGame['teamGameRank']>82).astype(np.int8)
    teamGame.loc[(teamGame['Season']==2012)&(teamGame['teamGameRank']>48),'Playoffs'] = 1 #fix for the lockout-shortened season
    teamGame.loc[(teamGame['Season']==2019)&(teamGame['Date']>'2020-03-12'),'Playoffs'] = 1 #fix for the first covid-shortened season
    teamGame.loc[(teamGame['Season']==2020)&(teamGame['teamGameRank']>56),'Playoffs'] = 1 #fix for the second covid-shortened season

    # add elo ratings
    teamGame = add_elo(teamGame)

    return teamGame

def add_scheduled_games(teamGame, schedule, season=2022):
    # adds scheduled games to end of teamGame, to create features for game predictions based on past games

    # do some processing in schedule dataframe first
    home = schedule[['game_id','date','home_team']].copy()
    home.columns = ['Game_Id','Date','Team']
    home['Home'] = 1
    away = schedule[['game_id','date','away_team']].copy()
    away.columns = ['Game_Id','Date','Team']
    away['Home'] = 0
    schedule = pd.concat([home, away], ignore_index=True)
    schedule['Team'] = schedule['Team'].replace({'PHX':'ARI', 'S.J':'SJS', 'L.A':'LAK', 'T.B':'TBL', 'N.J':'NJD'})
    schedule['Season'] = season
    schedule['DateInt'] = schedule['Date'].str.replace('-','').astype(np.int64)

    # combine dataframes
    teamGame = teamGame.loc[teamGame['Date']<(schedule['Date'].min())]
    teamGame = pd.concat([teamGame, schedule], ignore_index=True)

    # add elo ratings
    teamGame = add_elo(teamGame)

    return teamGame

def _add_lag(df, cols, lag, groupCol):
    new_cols = [col+'_last'+str(lag) for col in cols]
    df[new_cols] = df.groupby(groupCol)[cols].transform(lambda x: x.rolling(window=lag).mean())
    df[new_cols] = df.groupby(groupCol)[new_cols].shift(1)

    return df

def add_game_features(teamGame, playerGame, preseason_config_team_file='configs/preseason_config_team.json',
        tanh_inseason_coefs_team_file='configs/tanh_inseason_coefs_team.json'):
    # adds features needed for game predictions

    # read configs
    with open(preseason_config_team_file) as f:
        preseason_config_team = json.load(f)
    with open(tanh_inseason_coefs_team_file) as f:
        tanh_inseason_coefs_team = json.load(f)

    # add rookie preseason ratings
    playerGame.loc[(playerGame['Position']=='F')&(playerGame['preseason_xGC60_5v5'].isnull()), 'preseason_xGC60_5v5'] = 0.0179
    playerGame.loc[(playerGame['Position']=='F')&(playerGame['preseason_xGP60_5v5'].isnull()), 'preseason_xGP60_5v5'] = -0.0127
    playerGame.loc[(playerGame['Position']=='F')&(playerGame['preseason_xGC60_PP'].isnull()), 'preseason_xGC60_PP'] = -0.1067
    playerGame.loc[(playerGame['Position']=='F')&(playerGame['preseason_xGP60_PK'].isnull()), 'preseason_xGP60_PK'] = -0.5281
    playerGame.loc[(playerGame['Position']=='F')&(playerGame['preseason_xGI60_Pens'].isnull()), 'preseason_xGI60_Pens'] = -0.0048
    playerGame.loc[(playerGame['Position']=='D')&(playerGame['preseason_xGC60_5v5'].isnull()), 'preseason_xGC60_5v5'] = -0.0005
    playerGame.loc[(playerGame['Position']=='D')&(playerGame['preseason_xGP60_5v5'].isnull()), 'preseason_xGP60_5v5'] = 0.0337
    playerGame.loc[(playerGame['Position']=='D')&(playerGame['preseason_xGC60_PP'].isnull()), 'preseason_xGC60_PP'] = -0.1030
    playerGame.loc[(playerGame['Position']=='D')&(playerGame['preseason_xGP60_PK'].isnull()), 'preseason_xGP60_PK'] = -0.0717
    playerGame.loc[(playerGame['Position']=='D')&(playerGame['preseason_xGI60_Pens'].isnull()), 'preseason_xGI60_Pens'] = -0.0267
    playerGame.loc[(playerGame['Position']=='G')&(playerGame['preseason_xGI60'].isnull()), 'preseason_xGI60'] = -0.0147

    # fill first games of season with preseason expectations
    playerGame['PlayerGameNum'] = playerGame.groupby(['Team','Season'])['DateInt'].rank("dense")
    playerGame.loc[playerGame['PlayerGameNum']==1, 'xGC60_5v5'] = playerGame.loc[playerGame['PlayerGameNum']==1, 'preseason_xGC60_5v5']
    playerGame.loc[playerGame['PlayerGameNum']==1, 'xGP60_5v5'] = playerGame.loc[playerGame['PlayerGameNum']==1, 'preseason_xGP60_5v5']
    playerGame.loc[playerGame['PlayerGameNum']==1, 'xGC60_PP'] = playerGame.loc[playerGame['PlayerGameNum']==1, 'preseason_xGC60_PP']
    playerGame.loc[playerGame['PlayerGameNum']==1, 'xGP60_PK'] = playerGame.loc[playerGame['PlayerGameNum']==1, 'preseason_xGP60_PK']
    playerGame.loc[playerGame['PlayerGameNum']==1, 'xGI60_Pens'] = playerGame.loc[playerGame['PlayerGameNum']==1, 'preseason_xGI60_Pens']
    playerGame.loc[playerGame['PlayerGameNum']==1, 'xGI60'] = playerGame.loc[playerGame['PlayerGameNum']==1, 'preseason_xGI60']

    # project future player impacts based on combination of inseason performance and preseason expectations
    playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'xGC60_5v5'] = \
        (np.tanh((playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/115) \
        * playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'prevGames_GC60_5v5']) \
        + ((1 - np.tanh((playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/115)) \
        * playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'preseason_xGC60_5v5'])
    playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'xGP60_5v5'] = \
        (np.tanh((playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/148) \
        * playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'prevGames_GP60_5v5']) \
        + ((1 - np.tanh((playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/148)) \
        * playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'preseason_xGP60_5v5'])
    playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'xGC60_PP'] = \
        (np.tanh((playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/235) \
        * playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'prevGames_GC60_PP']) \
        + ((1 - np.tanh((playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/235)) \
        * playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'preseason_xGC60_PP'])
    playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'xGP60_PK'] = \
        (np.tanh((playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/121) \
        * playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'prevGames_GP60_PK']) \
        + ((1 - np.tanh((playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/121)) \
        * playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'preseason_xGP60_PK'])
    playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'xGI60_Pens'] = \
        (np.tanh((playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/151) \
        * playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'prevGames_GI60_Pens']) \
        + ((1 - np.tanh((playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/151)) \
        * playerGame.loc[(playerGame['Position']=='F')&(playerGame['PlayerGameNum']>1), 'preseason_xGI60_Pens'])

    playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'xGC60_5v5'] = \
        (np.tanh((playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/108) \
        * playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'prevGames_GC60_5v5']) \
        + ((1 - np.tanh((playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/108)) \
        * playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'preseason_xGC60_5v5'])
    playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'xGP60_5v5'] = \
        (np.tanh((playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/125) \
        * playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'prevGames_GP60_5v5']) \
        + ((1 - np.tanh((playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/125)) \
        * playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'preseason_xGP60_5v5'])
    playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'xGC60_PP'] = \
        (np.tanh((playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/194) \
        * playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'prevGames_GC60_PP']) \
        + ((1 - np.tanh((playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/194)) \
        * playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'preseason_xGC60_PP'])
    playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'xGP60_PK'] = \
        (np.tanh((playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/75) \
        * playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'prevGames_GP60_PK']) \
        + ((1 - np.tanh((playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/75)) \
        * playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'preseason_xGP60_PK'])
    playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'xGI60_Pens'] = \
        (np.tanh((playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/90) \
        * playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'prevGames_GI60_Pens']) \
        + ((1 - np.tanh((playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/90)) \
        * playerGame.loc[(playerGame['Position']=='D')&(playerGame['PlayerGameNum']>1), 'preseason_xGI60_Pens'])

    playerGame.loc[(playerGame['Position']=='G')&(playerGame['PlayerGameNum']>1), 'xGI60'] = \
        (np.tanh((playerGame.loc[(playerGame['Position']=='G')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/39) \
        * playerGame.loc[(playerGame['Position']=='G')&(playerGame['PlayerGameNum']>1), 'prevGames_GI60']) \
        + ((1 - np.tanh((playerGame.loc[(playerGame['Position']=='G')&(playerGame['PlayerGameNum']>1), 'PlayerGameNum']-1)/39)) \
        * playerGame.loc[(playerGame['Position']=='G')&(playerGame['PlayerGameNum']>1), 'preseason_xGI60'])

    # aggregate individual player projections into team features
    playerTeamGame = playerGame.loc[playerGame['Position'].isin(['F','D'])].groupby(['Date','Team']).agg({
        'xGC60_5v5' : ['sum','median','max','std'],
        'xGP60_5v5' : ['sum','median','max','std'],
        'xGI60_Pens' : ['sum','median','max','std']
    }).reset_index()
    playerTeamGame.columns = ['Date','Team','xGC60_5v5_sum','xGC60_5v5_median','xGC60_5v5_max','xGC60_5v5_std',
                              'xGP60_5v5_sum','xGP60_5v5_median','xGP60_5v5_max','xGP60_5v5_std',
                              'xGI60_Pens_sum','xGI60_Pens_median','xGI60_Pens_max','xGI60_Pens_std'
                              ]
    games_df = teamGame.merge(playerTeamGame, on=['Date','Team'], how='left')

    playerGame['PP_Rank'] = playerGame.groupby(['Date','Team','Position'])['xGC60_PP'].rank("dense", ascending=False)
    playerGame['PK_Rank'] = playerGame.groupby(['Date','Team','Position'])['xGP60_PK'].rank("dense", ascending=False)

    playerTeamGame = playerGame.loc[((playerGame['Position']=='F')&(playerGame['PP_Rank']<=6))|\
        ((playerGame['Position']=='D')&(playerGame['PP_Rank']<=2))].groupby(['Date','Team']).agg({
        'xGC60_PP' : ['sum','median','max','std']
    }).reset_index()
    playerTeamGame.columns = ['Date','Team','xGC60_PP_sum','xGC60_PP_median','xGC60_PP_max','xGC60_PP_std']
    games_df = games_df.merge(playerTeamGame, on=['Date','Team'], how='left')

    playerTeamGame = playerGame.loc[((playerGame['Position']=='F')&(playerGame['PK_Rank']<=4))|\
        ((playerGame['Position']=='D')&(playerGame['PP_Rank']<=4))].groupby(['Date','Team']).agg({
        'xGP60_PK' : ['sum','median','max','std']
    }).reset_index()
    playerTeamGame.columns = ['Date','Team','xGP60_PK_sum','xGP60_PK_median','xGP60_PK_max','xGP60_PK_std']
    games_df = games_df.merge(playerTeamGame, on=['Date','Team'], how='left')

    playerTeamGame = playerGame.loc[playerGame['Position']=='F'].groupby(['Date','Team']).agg({
        'xGC60_5v5' : ['sum','median','max','std'],
        'xGP60_5v5' : ['sum','median','max','std'],
        'xGI60_Pens' : ['sum','median','max','std']
    }).reset_index()
    playerTeamGame.columns = ['Date','Team','xGC60_5v5_F_sum','xGC60_5v5_F_median','xGC60_5v5_F_max','xGC60_5v5_F_std',
                              'xGP60_5v5_F_sum','xGP60_5v5_F_median','xGP60_5v5_F_max','xGP60_5v5_F_std',
                              'xGI60_Pens_F_sum','xGI60_Pens_F_median','xGI60_Pens_F_max','xGI60_Pens_F_std'
                              ]
    games_df = games_df.merge(playerTeamGame, on=['Date','Team'], how='left')

    playerTeamGame = playerGame.loc[playerGame['Position']=='D'].groupby(['Date','Team']).agg({
        'xGC60_5v5' : ['sum','median','max','std'],
        'xGP60_5v5' : ['sum','median','max','std'],
        'xGI60_Pens' : ['sum','median','max','std']
    }).reset_index()
    playerTeamGame.columns = ['Date','Team','xGC60_5v5_D_sum','xGC60_5v5_D_median','xGC60_5v5_D_max','xGC60_5v5_D_std',
                              'xGP60_5v5_D_sum','xGP60_5v5_D_median','xGP60_5v5_D_max','xGP60_5v5_D_std',
                              'xGI60_Pens_D_sum','xGI60_Pens_D_median','xGI60_Pens_D_max','xGI60_Pens_D_std'
                              ]
    games_df = games_df.merge(playerTeamGame, on=['Date','Team'], how='left')

    playerTeamGame = playerGame.loc[playerGame['Position']=='G'].groupby(['Date','Team','PlayerID']).agg({
        'xGI60' : ['mean']
    }).reset_index()
    playerTeamGame.columns = ['Date','Team','StartingGoalie_Id','Goalie_xGI60']
    games_df = games_df.merge(playerTeamGame, on=['Date','Team','StartingGoalie_Id'], how='left')

    # add some lag features (rolling averages of past games)
    metrics = ['Goals','Shots','ShotAttempts','UnblockedShotAttempts','xG','xG_flurry',
               'Goals_5v5','Shots_5v5','ShotAttempts_5v5','UnblockedShotAttempts_5v5','xG_5v5','xG_flurry_5v5',
               'GoalsAdjusted','ShotsAdjusted','ShotAttemptsAdjusted','UnblockedShotAttemptsAdjusted','xGAdjusted','xG_flurryAdjusted',
               'GoalsAdjusted_5v5','ShotsAdjusted_5v5','ShotAttemptsAdjusted_5v5','UnblockedShotAttemptsAdjusted_5v5','xGAdjusted_5v5','xG_flurryAdjusted_5v5']
    games_df = games_df.loc[games_df['Season']>=playerGame['Season'].max()-2]
    games_df = _add_lag(games_df, metrics, 8, 'Team')
    games_df = _add_lag(games_df, metrics, 16, 'Team')
    games_df = _add_lag(games_df, metrics, 32, 'Team')
    games_df = _add_lag(games_df, metrics, 64, 'Team')

    # do a self-join to get features for the opposing team
    games_df['teamGameVal'] = games_df.groupby(['Game_Id','Season','Date'])['xG'].rank("dense").replace(2, -1)
    games_df['LastGame'] = games_df.groupby('Team')['Date'].shift(1)
    games_df['RestDays'] = (pd.to_datetime(games_df['Date']) - pd.to_datetime(games_df['LastGame']))/np.timedelta64(1,'D')
    games_df['BackToBack'] = (games_df['RestDays']==1).astype(np.int16)
    games_df_opp = games_df.copy()
    games_df_opp['teamGameVal'] = games_df_opp['teamGameVal'] * -1
    games_df_opp = games_df_opp.drop(['Win','DateInt','Playoffs','Home','teamGameRankOverall','teamGameRank','StartingGoalie','StartingGoalie_Id'],1)
    for c in games_df_opp.columns.tolist():
        if c not in ['Game_Id','Date','Team','Season','teamGameVal']:
            games_df_opp = games_df_opp.rename(columns = {c : 'Opp_'+c})
    games_df_opp = games_df_opp.rename(columns = {'Team' : 'Opp'})
    games_df = games_df.merge(games_df_opp, on=['Game_Id','Date','Season','teamGameVal'])
    games_df = games_df.drop('teamGameVal',1)
    games_df['EloDiff'] = games_df['Elo'] - games_df['Opp_Elo']
    games_df['EloDiff_538adj'] = (games_df['Elo'] - games_df['Opp_Elo'] + (games_df['Home']*50) + ((games_df['Home']-1)*50))*(games_df['Playoffs']*0.25+1)

    for m in preseason_config_team.keys():
        # add new team preseason ratings
        games_df.loc[games_df['preseason_x'+m].isnull(), 'preseason_x'+m] = preseason_config_team[m][0][0]
        games_df.loc[games_df['teamGameRank']==1, 'x'+m] = games_df.loc[games_df['teamGameRank']==1, 'preseason_x'+m]

        # project future stats based on combination of inseason performance and preseason expectations
        games_df.loc[games_df['teamGameRank']>1, 'x'+m] = \
            (np.tanh((games_df.loc[games_df['teamGameRank']>1, 'teamGameRank']-1)/tanh_inseason_coefs_team[m]) \
            * games_df.loc[games_df['teamGameRank']>1, 'prevGames_'+m]) \
            + ((1 - np.tanh((games_df.loc[games_df['teamGameRank']>1, 'teamGameRank']-1)/tanh_inseason_coefs_team[m])) \
            * games_df.loc[games_df['teamGameRank']>1, 'preseason_x'+m])

        games_df = games_df.drop(['prevGames_'+m, 'preseason_x'+m], 1)

    # add lag features for allowed stats
    metrics = ['Opp_Goals','Opp_Shots','Opp_ShotAttempts',
            'Opp_UnblockedShotAttempts','Opp_xG','Opp_xG_flurry',
        'Opp_Goals_5v5','Opp_Shots_5v5','Opp_ShotAttempts_5v5',
            'Opp_UnblockedShotAttempts_5v5','Opp_xG_5v5','Opp_xG_flurry_5v5',
        'Opp_GoalsAdjusted','Opp_ShotsAdjusted','Opp_ShotAttemptsAdjusted',
            'Opp_UnblockedShotAttemptsAdjusted','Opp_xGAdjusted','Opp_xG_flurryAdjusted',
        'Opp_GoalsAdjusted_5v5','Opp_ShotsAdjusted_5v5','Opp_ShotAttemptsAdjusted_5v5',
            'Opp_UnblockedShotAttemptsAdjusted_5v5','Opp_xGAdjusted_5v5','Opp_xG_flurryAdjusted_5v5']
    games_df = _add_lag(games_df, metrics, 8, 'Team')
    games_df = _add_lag(games_df, metrics, 16, 'Team')
    games_df = _add_lag(games_df, metrics, 32, 'Team')
    games_df = _add_lag(games_df, metrics, 64, 'Team')
    # rename columns for clarity
    for m in metrics:
        games_df = games_df.rename(columns = {m+'_last8' : m[4:]+'Against_last8',
            m+'_last16' : m[4:]+'Against_last16', m+'_last32' : m[4:]+'Against_last32', m+'_last64' : m[4:]+'Against_last64'})

    # filter for only this season
    games_df = games_df.loc[games_df['Season']==playerGame['Season'].max()]

    # do a second self-join to get final features for the opposing team
    games_df_opp = games_df.copy()
    keep_cols = ['Game_Id','Date','Team','Season','xGoals','xShots','xShotAttempts','xUnblockedShotAttempts','xxG','xxG_flurry',
        'xGoalsAgainst','xShotsAgainst','xShotAttemptsAgainst','xUnblockedShotAttemptsAgainst','xxGAgainst','xxG_flurryAgainst']
    last8_cols = [c[4:]+'Against_last8' for c in metrics]
    last16_cols = [c[4:]+'Against_last16' for c in metrics]
    last32_cols = [c[4:]+'Against_last32' for c in metrics]
    last64_cols = [c[4:]+'Against_last64' for c in metrics]
    keep_cols = keep_cols + last8_cols + last16_cols + last32_cols + last64_cols
    games_df_opp = games_df_opp[keep_cols]
    for c in games_df_opp.columns.tolist():
        if c not in ['Game_Id','Date','Team','Season']:
            games_df_opp = games_df_opp.rename(columns = {c : 'Opp_'+c})
    games_df_opp = games_df_opp.rename(columns = {'Team' : 'Opp'})
    games_df = games_df.merge(games_df_opp, on=['Game_Id','Date','Season','Opp'])

    # home games only, and some final feature calculation
    games_df = games_df.loc[(games_df['Home']==1)&(games_df['Playoffs']==0)]
    games_df = games_df.loc[games_df['Season']>=2015]
    games_df['GI_5v5_Off_Diff'] = games_df['xGC60_5v5_sum'] - games_df['Opp_xGP60_5v5_sum']
    games_df['GI_5v5_Def_Diff'] = games_df['xGP60_5v5_sum'] - games_df['Opp_xGC60_5v5_sum']
    games_df['GI_5v5_Diff'] = games_df['GI_5v5_Off_Diff'] + games_df['GI_5v5_Def_Diff']
    games_df['GI_PP_Diff'] = games_df['xGC60_PP_sum'] - games_df['Opp_xGP60_PK_sum']
    games_df['GI_PK_Diff'] = games_df['xGP60_PK_sum'] - games_df['Opp_xGC60_PP_sum']

    return games_df
