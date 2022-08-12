import pandas as pd
import numpy as np
import os
import gc

def add_player_inseason_ratings(playerGame, toiOverlap):
    # add impact from previous games so far in the season to playerGame
    season = playerGame['Season'].unique()[0]

    xGs = playerGame.loc[playerGame['Playoffs']==0]
    xGs['PlayerGameNum'] = xGs.groupby(['Player','PlayerID'])['DateInt'].rank('dense')
    xGs = xGs.drop(['Playoffs','DateInt','Season','Date','Game_Id','Playoffs','DateInt','PlayerGameID','Team'],1)

    toiOverlap = toiOverlap.rename(columns={'Player_x':'Player', 'Player_Id_x':'PlayerID'})
    #toiOverlap['PlayerGameNum'] = toiOverlap.groupby(['Player','PlayerID'])['Game_Id'].rank('dense')
    toiOverlap_5v5 = toiOverlap.loc[toiOverlap['Strength']=='5x5']
    toiOverlap_PP = toiOverlap.loc[toiOverlap['Strength'].isin(['5x4','5x3','4x3'])]
    toiOverlap_PK = toiOverlap.loc[toiOverlap['Strength'].isin(['4x5','3x5','3x4'])]

    ratings = toiOverlap.copy(deep=True)
    ratings = ratings.loc[ratings['Game_Id'].isin(xGs['Game_Id'].unique())]
    xGs = xGs.drop(columns=['Playoffs','DateInt','Season','Date','Game_Id','Playoffs','DateInt','PlayerGameID','Team'])
    xGs = xGs.fillna(0).groupby(['Player','PlayerID','Position'], as_index=False).sum()
    xGs = xGs.loc[xGs['TOI_5v5']>0]
    xGs['OZoneStartRate_5v5'] = xGs['OZoneStartCount_5v5']/(xGs['OZoneStartCount_5v5']+xGs['NZoneStartCount_5v5']+xGs['DZoneStartCount_5v5'])
    xGs['DZoneStartRate_5v5'] = xGs['DZoneStartCount_5v5']/(xGs['OZoneStartCount_5v5']+xGs['NZoneStartCount_5v5']+xGs['DZoneStartCount_5v5'])
    xGs['metric_O'] = 0.
    xGs.loc[xGs['Position']=='F', 'metric_O'] = 3600*(
        .163*xGs.loc[xGs['Position']=='F', 'Goals_5v5_onice']
        + .185*.091286*xGs.loc[xGs['Position']=='F', 'ShotsAdjusted_5v5_onice']
        + .262*xGs.loc[xGs['Position']=='F', 'xG_flurry_5v5_onice']
        )/(xGs.loc[xGs['Position']=='F', 'TOI_5v5'] * (.163+.185+.262))
    xGs.loc[xGs['Position']=='D', 'metric_O'] = 3600*(
        .064*xGs.loc[xGs['Position']=='D', 'GoalsAdjusted_5v5_onice']
        + .132*.049399*xGs.loc[xGs['Position']=='D', 'ShotAttemptsAdjusted_5v5_onice']
        + .160*xGs.loc[xGs['Position']=='D', 'xG_5v5_onice']
        )/(xGs.loc[xGs['Position']=='D', 'TOI_5v5'] * (.064+.132+.160))

    xGs['metric_D'] = 0.
    xGs.loc[xGs['Position']=='F', 'metric_D'] = 3600*(
        .031*xGs.loc[xGs['Position']=='F', 'GoalsAgainst_5v5_onice']
        + .140*.065956*xGs.loc[xGs['Position']=='F', 'UnblockedShotAttemptsAdjustedAgainst_5v5_onice']
        + .175*xGs.loc[xGs['Position']=='F', 'xG_flurryAdjustedAgainst_5v5_onice']
        )/(xGs.loc[xGs['Position']=='F', 'TOI_5v5'] * (.031+.140+.175))
    xGs.loc[xGs['Position']=='D', 'metric_D'] = 3600*(
        .041*xGs.loc[xGs['Position']=='D', 'GoalsAgainst_5v5_onice']
        + .172*.065956*xGs.loc[xGs['Position']=='D', 'UnblockedShotAttemptsAdjustedAgainst_5v5_onice']
        + .216*xGs.loc[xGs['Position']=='D', 'xGAdjustedAgainst_5v5_onice']
        )/(xGs.loc[xGs['Position']=='D', 'TOI_5v5'] * (.041+.172+.216))

    xGs['metric_PP'] = 0.
    xGs.loc[xGs['Position']=='F', 'metric_PP'] = 3600*(
        .154*xGs.loc[xGs['Position']=='F', 'GoalsAdjusted_PP_onice']
        + .268*.065956*xGs.loc[xGs['Position']=='F', 'UnblockedShotAttempts_PP_onice']
        + .293*xGs.loc[xGs['Position']=='F', 'xG_flurry_PP_onice']
        )/(xGs.loc[xGs['Position']=='F', 'TOI_PP'] * (.154+.268+.293))
    xGs.loc[xGs['Position']=='D', 'metric_PP'] = 3600*(
        .104*xGs.loc[xGs['Position']=='D', 'GoalsAdjusted_PP_onice']
        + .249*.065956*xGs.loc[xGs['Position']=='D', 'UnblockedShotAttempts_PP_onice']
        + .215*xGs.loc[xGs['Position']=='D', 'xG_flurry_PP_onice']
        )/(xGs.loc[xGs['Position']=='D', 'TOI_PP'] * (.104+.249+.215))

    xGs['metric_PK'] = 0.
    xGs.loc[xGs['Position']=='F', 'metric_PK'] = 3600*(
        .019*xGs.loc[xGs['Position']=='F', 'GoalsAdjustedAgainst_PK_onice']
        + .155*.049399*xGs.loc[xGs['Position']=='F', 'ShotAttemptsAgainst_PK_onice']
        + .097*xGs.loc[xGs['Position']=='F', 'xG_flurryAdjustedAgainst_PK_onice']
        )/(xGs.loc[xGs['Position']=='F', 'TOI_PK'] * (.019+.155+.097))
    xGs.loc[xGs['Position']=='D', 'metric_PK'] = 3600*(
        .017*xGs.loc[xGs['Position']=='D', 'GoalsAgainst_PK_onice']
        + .166*.049399*xGs.loc[xGs['Position']=='D', 'ShotAttemptsAgainst_PK_onice']
        + .065*xGs.loc[xGs['Position']=='D', 'xG_flurryAgainst_PK_onice']
        )/(xGs.loc[xGs['Position']=='D', 'TOI_PK'] * (.017+.166+.065))

    xGs['metric_G'] = 3600*(xGs['xGAgainst_onice']-xGs['GoalsAgainst_onice'])/xGs['TOI']

    # loop through one value at a time for PlayerGameNum
    inseason_ratings = pd.DataFrame()
    for i in xGs['Date'].sort_values().unique()[1:]:
        xGs_temp = xGs.loc[xGs['Date']<i]
        ratings_temp = ratings.loc[ratings['Game_Id'].isin(xGs_temp['Game_Id'].unique())]

        #compute mean metrics
        metric_mean_F_O = 3600*(
            .163*xGs_temp.loc[xGs_temp['Position']=='F', 'Goals_5v5_onice'].sum()
            + .185*.091286*xGs_temp.loc[xGs_temp['Position']=='F', 'ShotsAdjusted_5v5_onice'].sum()
            + .262*xGs_temp.loc[xGs_temp['Position']=='F', 'xG_flurry_5v5_onice'].sum()
            )/(xGs_temp.loc[xGs_temp['Position']=='F', 'TOI_5v5'].sum() * (.163+.185+.262))
        metric_mean_D_O = 3600*(
            .064*xGs_temp.loc[xGs_temp['Position']=='D', 'GoalsAdjusted_5v5_onice'].sum()
            + .132*.049399*xGs_temp.loc[xGs_temp['Position']=='D', 'ShotAttemptsAdjusted_5v5_onice'].sum()
            + .160*xGs_temp.loc[xGs_temp['Position']=='D', 'xG_5v5_onice'].sum()
            )/(xGs_temp.loc[xGs_temp['Position']=='D', 'TOI_5v5'].sum() * (.064+.132+.160))
        metric_mean_F_D = 3600*(
            .031*xGs_temp.loc[xGs_temp['Position']=='F', 'GoalsAgainst_5v5_onice'].sum()
            + .140*.065956*xGs_temp.loc[xGs_temp['Position']=='F', 'UnblockedShotAttemptsAdjustedAgainst_5v5_onice'].sum()
            + .175*xGs_temp.loc[xGs_temp['Position']=='F', 'xG_flurryAdjustedAgainst_5v5_onice'].sum()
            )/(xGs_temp.loc[xGs_temp['Position']=='F', 'TOI_5v5'].sum() * (.031+.140+.175))
        metric_mean_D_D = 3600*(
            .041*xGs_temp.loc[xGs_temp['Position']=='D', 'GoalsAgainst_5v5_onice'].sum()
            + .172*.065956*xGs_temp.loc[xGs_temp['Position']=='D', 'UnblockedShotAttemptsAdjustedAgainst_5v5_onice'].sum()
            + .216*xGs_temp.loc[xGs_temp['Position']=='D', 'xGAdjustedAgainst_5v5_onice'].sum()
            )/(xGs_temp.loc[xGs_temp['Position']=='D', 'TOI_5v5'].sum() * (.041+.172+.216))
        metric_mean_F_PP = 3600*(
            .154*xGs_temp.loc[xGs_temp['Position']=='F', 'GoalsAdjusted_PP_onice'].sum()
            + .268*.065956*xGs_temp.loc[xGs_temp['Position']=='F', 'UnblockedShotAttempts_PP_onice'].sum()
            + .293*xGs_temp.loc[xGs_temp['Position']=='F', 'xG_flurry_PP_onice'].sum()
            )/(xGs_temp.loc[xGs_temp['Position']=='F', 'TOI_PP'].sum() * (.154+.268+.293))
        metric_mean_D_PP = 3600*(
            .104*xGs_temp.loc[xGs_temp['Position']=='D', 'GoalsAdjusted_PP_onice'].sum()
            + .249*.065956*xGs_temp.loc[xGs_temp['Position']=='D', 'UnblockedShotAttempts_PP_onice'].sum()
            + .215*xGs_temp.loc[xGs_temp['Position']=='D', 'xG_flurry_PP_onice'].sum()
            )/(xGs_temp.loc[xGs_temp['Position']=='D', 'TOI_PP'].sum() * (.104+.249+.215))
        metric_mean_F_PK = 3600*(
            .019*xGs_temp.loc[xGs_temp['Position']=='F', 'GoalsAdjustedAgainst_PK_onice'].sum()
            + .155*.049399*xGs_temp.loc[xGs_temp['Position']=='F', 'ShotAttemptsAgainst_PK_onice'].sum()
            + .097*xGs_temp.loc[xGs_temp['Position']=='F', 'xG_flurryAdjustedAgainst_PK_onice'].sum()
            )/(xGs_temp.loc[xGs_temp['Position']=='F', 'TOI_PK'].sum() * (.019+.155+.097))
        metric_mean_D_PK = 3600*(
            .017*xGs_temp.loc[xGs_temp['Position']=='D', 'GoalsAgainst_PK_onice'].sum()
            + .166*.049399*xGs_temp.loc[xGs_temp['Position']=='D', 'ShotAttemptsAgainst_PK_onice'].sum()
            + .065*xGs_temp.loc[xGs_temp['Position']=='D', 'xG_flurryAgainst_PK_onice'].sum()
            )/(xGs_temp.loc[xGs_temp['Position']=='D', 'TOI_PK'].sum() * (.017+.166+.065))

        ratings_PP = ratings_temp.loc[ratings['Strength'].isin(['5x4','5x3','4x3'])]
        ratings_PP = ratings_PP[['Player','PlayerID','Player_y','Player_Id_y','SameTeam','Overlap']]
        ratings_PP = ratings_PP.groupby(['Player','PlayerID','Player_y','Player_Id_y','SameTeam'], as_index=False).sum()
        ratings_PP = ratings_PP.merge(xGs_temp, on=['Player','PlayerID'])
        xG_others = xGs_temp.rename(columns={'Player':'Player_y', 'PlayerID':'Player_Id_y'})
        ratings_PP = ratings_PP.merge(xG_others, on=['Player_y','Player_Id_y'])
        del xG_others

        ratings_PK = ratings_temp.loc[ratings['Strength'].isin(['4x5','3x5','3x4'])]
        ratings_PK = ratings_PK[['Player','PlayerID','Player_y','Player_Id_y','SameTeam','Overlap']]
        ratings_PK = ratings_PK.groupby(['Player','PlayerID','Player_y','Player_Id_y','SameTeam'], as_index=False).sum()
        ratings_PK = ratings_PK.merge(xGs_temp, on=['Player','PlayerID'])
        xG_others = xGs_temp.rename(columns={'Player':'Player_y', 'PlayerID':'Player_Id_y'})
        ratings_PK = ratings_PK.merge(xG_others, on=['Player_y','Player_Id_y'])
        del xG_others

        ratings_temp = ratings_temp.loc[ratings['Strength']=='5x5']
        ratings_temp = ratings_temp[['Player','PlayerID','Player_y','Player_Id_y','SameTeam','Overlap']]
        ratings_temp = ratings_temp.groupby(['Player','PlayerID','Player_y','Player_Id_y','SameTeam'], as_index=False).sum()
        ratings_temp = ratings_temp.merge(xGs_temp, on=['Player','PlayerID'])
        xG_others = xGs_temp.rename(columns={'Player':'Player_y', 'PlayerID':'Player_Id_y'})
        ratings_temp = ratings_temp.merge(xG_others, on=['Player_y','Player_Id_y'])
        del xG_others

        # F, 5v5
        ratings_F = ratings_temp.loc[ratings['Position_x']=='F']
        ratings_F = ratings_F.loc[ratings_F['Position_y']!='G']
        ratings_F['metricSum_O_team'] = 0.
        ratings_F.loc[ratings_F['SameTeam'], 'metricSum_O_team'] = ratings_F.loc[ratings_F['SameTeam'], 'metric_O_y']*ratings_F.loc[ratings_F['SameTeam'], 'Overlap']/\
            (ratings_F.loc[ratings_F['SameTeam'], 'TOI_5v5_x']*4)
        ratings_F['metricSum_O_comp'] = 0.
        ratings_F.loc[~ratings_F['SameTeam'], 'metricSum_O_comp'] = ratings_F.loc[~ratings_F['SameTeam'], 'metric_D_y']*ratings_F.loc[~ratings_F['SameTeam'], 'Overlap']/\
            (ratings_F.loc[~ratings_F['SameTeam'], 'TOI_5v5_x']*5)
        ratings_F['metricSum_D_team'] = 0.
        ratings_F.loc[ratings_F['SameTeam'], 'metricSum_D_team'] = ratings_F.loc[ratings_F['SameTeam'], 'metric_D_y']*ratings_F.loc[ratings_F['SameTeam'], 'Overlap']/\
            (ratings_F.loc[ratings_F['SameTeam'], 'TOI_5v5_x']*4)
        ratings_F['metricSum_D_comp'] = 0.
        ratings_F.loc[~ratings_F['SameTeam'], 'metricSum_D_comp'] = ratings_F.loc[~ratings_F['SameTeam'], 'metric_O_y']*ratings_F.loc[~ratings_F['SameTeam'], 'Overlap']/\
            (ratings_F.loc[~ratings_F['SameTeam'], 'TOI_5v5_x']*5)
        ratings_F = ratings_F.groupby(['Player','PlayerID','Position_x'], as_index=False).agg({
            'TOI_5v5_x' : 'max',
            'OZoneStartRate_5v5_x' : 'max',
            'DZoneStartRate_5v5_x' : 'max',
            'metric_O_x' : 'max',
            'metricSum_O_team' : 'sum',
            'metricSum_O_comp' : 'sum',
            'metric_D_x' : 'max',
            'metricSum_D_team' : 'sum',
            'metricSum_D_comp' : 'sum',
            'xG_5v5_onice_x' : 'max',
            'Goals_5v5_x' : 'max',
            'Shots_5v5_x' : 'max',
            'ShotAttempts_5v5_x' : 'max',
            'UnblockedShotAttempts_5v5_x' : 'max',
            'xG_5v5_x' : 'max',
            'xG_flurry_5v5_x' : 'max',
            'PrimaryAssists_5v5_x' : 'max',
            'SecondaryAssists_5v5_x' : 'max',
            'TOI_x' : 'max'
        })

        features = ['OZoneStartRate_5v5_x','metricSum_O_team','metricSum_O_comp']
        X = ratings_F.dropna(subset=features)[features].values
        Y = ratings_F.dropna(subset=features)['metric_O_x'].values
        model = LinearRegression()
        model.fit(X, Y)
        ratings_F['pred_metric_O_x'] = model.predict(ratings_F[features].fillna(0.4).values)
        ratings_F['metric_O_aboveExp'] = ratings_F['metric_O_x']-ratings_F['pred_metric_O_x']
        ratings_F['metric_O_aboveAvg'] = ratings_F['metric_O_x']-metric_mean_F_O
        ratings_F['indiv_contrib_5v5'] = (.142*ratings_F['Goals_5v5_x']+.114*ratings_F['PrimaryAssists_5v5_x']+.036*ratings_F['SecondaryAssists_5v5_x']+\
            .559*.049399*ratings_F['ShotAttempts_5v5_x']+\
            .374*ratings_F['xG_flurry_5v5_x'])*3600/(ratings_F['TOI_5v5_x']*(.142+.114+.036+.559+.374))
        indiv_contrib_mean = (ratings_F['indiv_contrib_5v5']*ratings_F['TOI_5v5_x']).sum()/ratings_F['TOI_5v5_x'].sum()
        ratings_F['indiv_contrib_5v5_aboveAvg'] = ratings_F['indiv_contrib_5v5']-indiv_contrib_mean
        ratings_F['GC60_5v5'] = (ratings_F['metric_O_aboveExp']+ratings_F['metric_O_aboveAvg']+ratings_F['indiv_contrib_5v5_aboveAvg'])/3
        ratings_F = ratings_F.rename(columns = {'Position_x':'Position', 'TOI_5v5_x':'TOI_5v5'})
        inseason_ratings_season = ratings_F[['Player','PlayerID','Position','TOI_5v5','GC60_5v5']]

        features = ['DZoneStartRate_5v5_x','metricSum_D_team','metricSum_D_comp']
        X = ratings_F.dropna(subset=features)[features].values
        Y = ratings_F.dropna(subset=features)['metric_D_x'].values
        model = LinearRegression()
        model.fit(X, Y)
        ratings_F['pred_metric_D_x'] = model.predict(ratings_F[features].fillna(0.4).values)
        ratings_F['metric_D_aboveExp'] = ratings_F['metric_D_x']-ratings_F['pred_metric_D_x']
        ratings_F['metric_D_aboveAvg'] = ratings_F['metric_D_x']-metric_mean_F_D
        ratings_F['GP60_5v5'] = (ratings_F['metric_D_aboveExp']+ratings_F['metric_D_aboveAvg'])/-2
        inseason_ratings_season = inseason_ratings_season.merge(ratings_F[['Player','PlayerID','Position','GP60_5v5']], \
            on=['Player','PlayerID','Position'])

        # F, PP
        ratings_PP_F = ratings_PP.loc[ratings_PP['Position_x']=='F']
        ratings_PP_F = ratings_PP_F.loc[ratings_PP_F['Position_y']!='G']
        ratings_PP_F = ratings_PP_F.loc[ratings_PP_F['Overlap']>0]
        ratings_PP_F = ratings_PP_F.loc[ratings_PP_F['TOI_PP_x']>0]
        ratings_PP_F = ratings_PP_F.loc[((ratings_PP_F['TOI_PK_y']>0)&(~ratings_PP_F['SameTeam']))|((ratings_PP_F['TOI_PP_y']>0)&(ratings_PP_F['SameTeam']))]
        ratings_PP_F['metricSum_PP_team'] = 0.
        ratings_PP_F.loc[ratings_PP_F['SameTeam'], 'metricSum_PP_team'] = ratings_PP_F.loc[ratings_PP_F['SameTeam'], 'metric_PP_y']*ratings_PP_F.loc[ratings_PP_F['SameTeam'], \
            'Overlap']/(ratings_PP_F.loc[ratings_PP_F['SameTeam'], 'TOI_PP_x']*4)
        ratings_PP_F['metricSum_PP_comp'] = 0.
        ratings_PP_F.loc[~ratings_PP_F['SameTeam'], 'metricSum_PP_comp'] = ratings_PP_F.loc[~ratings_PP_F['SameTeam'], 'metric_PK_y'].fillna(0)\
            *ratings_PP_F.loc[~ratings_PP_F['SameTeam'], 'Overlap']/(ratings_PP_F.loc[~ratings_PP_F['SameTeam'], 'TOI_PP_x']*4)
        ratings_PP_F = ratings_PP_F.groupby(['Player','PlayerID'], as_index=False).agg({
            'TOI_PP_x' : 'max',
            'metric_PP_x' : 'max',
            'metricSum_PP_team' : 'sum',
            'metricSum_PP_comp' : 'sum'
        })
        ratings_PP_F.columns = ['Player','PlayerID','TOI_PP','metric_O_PP','metricSum_team_PP','metricSum_comp_PP']

        features = ['metricSum_team_PP','metricSum_comp_PP']
        X = ratings_PP_F[features].values
        Y = ratings_PP_F['metric_O_PP'].values
        model = LinearRegression()
        model.fit(X, Y)
        intercept_F_GC_PP = model.intercept_
        weights_F_GC_PP = model.coef_

        # F, PP
        ratings_PP_F = ratings_PP.loc[ratings_PP['Position_x']=='F']
        ratings_PP_F = ratings_PP_F.loc[ratings_PP_F['Position_y']!='G']
        ratings_PP_F = ratings_PP_F.loc[ratings_PP_F['Overlap']>0]
        ratings_PP_F = ratings_PP_F.loc[ratings_PP_F['TOI_PP_x']>0]
        ratings_PP_F = ratings_PP_F.loc[((ratings_PP_F['TOI_PK_y']>0)&(~ratings_PP_F['SameTeam']))|((ratings_PP_F['TOI_PP_y']>0)&(ratings_PP_F['SameTeam']))]
        ratings_PP_F['metricSum_PP_team'] = 0.
        ratings_PP_F.loc[ratings_PP_F['SameTeam'], 'metricSum_PP_team'] = ratings_PP_F.loc[ratings_PP_F['SameTeam'], 'metric_PP_y']*ratings_PP_F.loc[ratings_PP_F['SameTeam'], \
            'Overlap']/(ratings_PP_F.loc[ratings_PP_F['SameTeam'], 'TOI_PP_x']*4)
        ratings_PP_F['metricSum_PP_comp'] = 0.
        ratings_PP_F.loc[~ratings_PP_F['SameTeam'], 'metricSum_PP_comp'] = ratings_PP_F.loc[~ratings_PP_F['SameTeam'], 'metric_PK_y'].fillna(0)\
            *ratings_PP_F.loc[~ratings_PP_F['SameTeam'], 'Overlap']/(ratings_PP_F.loc[~ratings_PP_F['SameTeam'], 'TOI_PP_x']*4)
        ratings_PP_F = ratings_PP_F.groupby(['Player','PlayerID'], as_index=False).agg({
            'TOI_PP_x' : 'max',
            'metric_PP_x' : 'max',
            'metricSum_PP_team' : 'sum',
            'metricSum_PP_comp' : 'sum',
            'xG_PP_onice_x' : 'max',
            'Goals_PP_x' : 'max',
            'Shots_PP_x' : 'max',
            'ShotAttempts_PP_x' : 'max',
            'UnblockedShotAttempts_PP_x' : 'max',
            'xG_PP_x' : 'max',
            'xG_flurry_PP_x' : 'max',
            'PrimaryAssists_PP_x' : 'max',
            'SecondaryAssists_PP_x' : 'max'
        })
        ratings_PP_F.columns = ['Player','PlayerID','Position_x','TOI_PP','metric_O_PP','metricSum_team_PP','metricSum_comp_PP','xG_PP_onice','Goals_PP',
            'Shots_PP','ShotAttempts_PP','UnblockedShotAttempts_PP','xG_PP','xG_flurry_PP','PrimaryAssists_PP','SecondaryAssists_PP']

        features = ['metricSum_team_PP','metricSum_comp_PP']
        X = ratings_PP_F[features].values
        Y = ratings_PP_F['metric_O_PP'].values
        model = LinearRegression()
        model.fit(X, Y)
        ratings_PP_F['pred_metric_O_PP'] = model.predict(X)
        ratings_PP_F['metric_PP_aboveExp'] = ratings_PP_F['metric_O_PP']-ratings_PP_F['pred_metric_O_PP']
        ratings_PP_F['metric_PP_aboveAvg'] = ratings_PP_F['metric_O_PP']-metric_mean_F_PP
        ratings_PP_F['indiv_contrib_PP'] = (.072*ratings_PP_F['Goals_PP']+.160*ratings_PP_F['PrimaryAssists_PP']+.091*ratings_PP_F['SecondaryAssists_PP']+\
            .549*.049399*ratings_PP_F['ShotAttempts_PP']+\
            .315*ratings_PP_F['xG_flurry_PP'])*3600/(ratings_PP_F['TOI_PP']*(.072+.160+.091+.549+.315))
        indiv_contrib_mean = (ratings_PP_F['indiv_contrib_PP']*ratings_PP_F['TOI_PP']).sum()/ratings_PP_F['TOI_PP'].sum()
        ratings_PP_F['indiv_contrib_PP_aboveAvg'] = ratings_PP_F['indiv_contrib_PP']-indiv_contrib_mean
        ratings_PP_F['GC60_PP'] = (ratings_PP_F['metric_PP_aboveExp']+ratings_PP_F['metric_PP_aboveAvg']+ratings_PP_F['indiv_contrib_PP_aboveAvg'])/3
        ratings_PP_F = ratings_PP_F.rename(columns = {'Position_x':'Position', 'TOI_PP_x':'TOI_PP'})
        inseason_ratings_season = inseason_ratings_season.merge(ratings_PP_F[['Player','PlayerID','Position','TOI_PP','GC60_PP']], \
            on=['Player','PlayerID','Position'], how='left')

        # F, PK
        ratings_PK_F = ratings_PK.loc[ratings_PK['Position_x']=='F']
        ratings_PK_F = ratings_PK_F.loc[ratings_PK_F['Position_y']!='G']
        ratings_PK_F = ratings_PK_F.loc[ratings_PK_F['Overlap']>0]
        ratings_PK_F = ratings_PK_F.loc[ratings_PK_F['TOI_PK_x']>0]
        ratings_PK_F = ratings_PK_F.loc[((ratings_PK_F['TOI_PP_y']>0)&(~ratings_PK_F['SameTeam']))|((ratings_PK_F['TOI_PK_y']>0)&(ratings_PK_F['SameTeam']))]
        ratings_PK_F['metricSum_PK_team'] = 0.
        ratings_PK_F.loc[ratings_PK_F['SameTeam'], 'metricSum_PK_team'] = ratings_PK_F.loc[ratings_PK_F['SameTeam'], 'metric_PK_y'].fillna(0)\
            *ratings_PK_F.loc[ratings_PK_F['SameTeam'], 'Overlap']/(ratings_PK_F.loc[ratings_PK_F['SameTeam'], 'TOI_PK_x']*3)
        ratings_PK_F['metricSum_PK_comp'] = 0.
        ratings_PK_F.loc[~ratings_PK_F['SameTeam'], 'metricSum_PK_comp'] = ratings_PK_F.loc[~ratings_PK_F['SameTeam'], 'metric_PP_y'].fillna(0)\
            *ratings_PK_F.loc[~ratings_PK_F['SameTeam'], 'Overlap']/(ratings_PK_F.loc[~ratings_PK_F['SameTeam'], 'TOI_PK_x']*5)
        ratings_PK_F = ratings_PK_F.groupby(['Player','PlayerID'], as_index=False).agg({
            'TOI_PK_x' : 'max',
            'metric_PK_x' : 'max',
            'metricSum_PK_team' : 'sum',
            'metricSum_PK_comp' : 'sum'
        })
        ratings_PK_F.columns = ['Player','PlayerID','Position_x','TOI_PK','metric_D_PK','metricSum_team_PK','metricSum_comp_PK']

        features = ['metricSum_team_PK','metricSum_comp_PK']
        X = ratings_PK_F[features].values
        Y = ratings_PK_F['metric_D_PK'].values
        model = LinearRegression()
        model.fit(X, Y)
        intercept_F_GP_PK = model.intercept_
        weights_F_GP_PK = model.coef_
        ratings_PK_F['pred_metric_D_PK'] = model.predict(X)
        ratings_PK_F['metric_D_aboveExp'] = ratings_PK_F['metric_D_PK']-ratings_PK_F['pred_metric_D_PK']
        ratings_PK_F['metric_D_aboveAvg'] = ratings_PK_F['metric_D_PK']-metric_mean_F_PK
        ratings_PK_F['GP60_PK'] = (ratings_PK_F['metric_D_aboveExp']+ratings_PK_F['metric_D_aboveAvg'])/-2
        ratings_PK_F = ratings_PK_F.rename(columns = {'Position_x':'Position', 'TOI_PK_x':'TOI_PK'})
        inseason_ratings_season = inseason_ratings_season.merge(ratings_PK_F[['Player','PlayerID','Position','TOI_PK','GP60_PK']], \
            on=['Player','PlayerID','Position'], how='left')

        # F, Pens
        ratings_pen_F = xGs_temp.loc[xGs_temp['Position']=='F', ['Player','PlayerID','Position','Penalties','PenaltiesDrawn','TOI']]
        ratings_pen_F = ratings_pen_F.groupby(['Player','PlayerID','Position'], as_index=False).sum()
        pen_val = ((3600*xGs_temp['Goals_PP_onice'].sum()/xGs_temp['TOI_PP'].sum()) - (3600*xGs_temp['Goals_PK_onice'].sum()/xGs_temp['TOI_PK'].sum()))*(2/60)
        ratings_pen_F['GI60_Pens'] = pen_val*3600*(.837*ratings_pen_F['PenaltiesDrawn'].fillna(0) - 1.163*ratings_pen_F['Penalties'].fillna(0))/ratings_pen_F['TOI']
        inseason_ratings_season = inseason_ratings_season.merge(ratings_pen_F[['Player','PlayerID','Position','TOI','GI60_Pens']], \
            on=['Player','PlayerID','Position'], how='left')

        # D, 5v5
        ratings_D = ratings_temp.loc[ratings['Position_x']=='D']
        ratings_D = ratings_D.loc[ratings_D['Position_y']!='G']
        ratings_D['metricSum_O_team'] = 0.
        ratings_D.loc[ratings_D['SameTeam'], 'metricSum_O_team'] = ratings_D.loc[ratings_D['SameTeam'], 'metric_O_y']*ratings_D.loc[ratings_D['SameTeam'], 'Overlap']/\
            (ratings_D.loc[ratings_D['SameTeam'], 'TOI_5v5_x']*4)
        ratings_D['metricSum_O_comp'] = 0.
        ratings_D.loc[~ratings_D['SameTeam'], 'metricSum_O_comp'] = ratings_D.loc[~ratings_D['SameTeam'], 'metric_D_y']*ratings_D.loc[~ratings_D['SameTeam'], 'Overlap']/\
            (ratings_D.loc[~ratings_D['SameTeam'], 'TOI_5v5_x']*5)
        ratings_D['metricSum_D_team'] = 0.
        ratings_D.loc[ratings_D['SameTeam'], 'metricSum_D_team'] = ratings_D.loc[ratings_D['SameTeam'], 'metric_D_y']*ratings_D.loc[ratings_D['SameTeam'], 'Overlap']/\
            (ratings_D.loc[ratings_D['SameTeam'], 'TOI_5v5_x']*4)
        ratings_D['metricSum_D_comp'] = 0.
        ratings_D.loc[~ratings_D['SameTeam'], 'metricSum_D_comp'] = ratings_D.loc[~ratings_D['SameTeam'], 'metric_O_y']*ratings_D.loc[~ratings_D['SameTeam'], 'Overlap']/\
            (ratings_D.loc[~ratings_D['SameTeam'], 'TOI_5v5_x']*5)
        ratings_D = ratings_D.groupby(['Player','PlayerID','Position_x'], as_index=False).agg({
            'TOI_5v5_x' : 'max',
            'OZoneStartRate_5v5_x' : 'max',
            'DZoneStartRate_5v5_x' : 'max',
            'metric_O_x' : 'max',
            'metricSum_O_team' : 'sum',
            'metricSum_O_comp' : 'sum',
            'metric_D_x' : 'max',
            'metricSum_D_team' : 'sum',
            'metricSum_D_comp' : 'sum',
            'xG_5v5_onice_x' : 'max',
            'Goals_5v5_x' : 'max',
            'Shots_5v5_x' : 'max',
            'ShotAttempts_5v5_x' : 'max',
            'UnblockedShotAttempts_5v5_x' : 'max',
            'xG_5v5_x' : 'max',
            'xG_flurry_5v5_x' : 'max',
            'PrimaryAssists_5v5_x' : 'max',
            'SecondaryAssists_5v5_x' : 'max',
            'TOI_x' : 'max'
        })

        features = ['OZoneStartRate_5v5_x','metricSum_O_team','metricSum_O_comp']
        X = ratings_D.dropna(subset=features)[features].values
        Y = ratings_D.dropna(subset=features)['metric_O_x'].values
        model = LinearRegression()
        model.fit(X, Y)
        ratings_D['pred_metric_O_x'] = model.predict(ratings_D[features].fillna(0.4).values)
        ratings_D['metric_O_aboveExp'] = ratings_D['metric_O_x']-ratings_D['pred_metric_O_x']
        ratings_D['metric_O_aboveAvg'] = ratings_D['metric_O_x']-metric_mean_D_O
        ratings_D['indiv_contrib_5v5'] = (.070*ratings_D['Goals_5v5_x']+.050*ratings_D['PrimaryAssists_5v5_x']+.021*ratings_D['SecondaryAssists_5v5_x']+\
            .502*.049399*ratings_D['ShotAttempts_5v5_x']+\
            .386*ratings_D['xG_flurry_5v5_x'])*3600/(ratings_D['TOI_5v5_x']*(.070+.050+.021+.502+.386))
        indiv_contrib_mean = (ratings_D['indiv_contrib_5v5']*ratings_D['TOI_5v5_x']).sum()/ratings_D['TOI_5v5_x'].sum()
        ratings_D['indiv_contrib_5v5_aboveAvg'] = ratings_D['indiv_contrib_5v5']-indiv_contrib_mean
        ratings_D['GC60_5v5'] = (ratings_D['metric_O_aboveExp']+ratings_D['metric_O_aboveAvg']+ratings_D['indiv_contrib_5v5_aboveAvg'])/3
        ratings_D = ratings_D.rename(columns = {'Position_x':'Position', 'TOI_5v5_x':'TOI_5v5'})
        inseason_ratings_season_D = ratings_D[['Player','PlayerID','Position','TOI_5v5','GC60_5v5']]

        features = ['DZoneStartRate_5v5_x','metricSum_D_team','metricSum_D_comp']
        X = ratings_D.dropna(subset=features)[features].values
        Y = ratings_D.dropna(subset=features)['metric_D_x'].values
        model = LinearRegression()
        model.fit(X, Y)
        ratings_D['pred_metric_D_x'] = model.predict(ratings_D[features].fillna(0.4).values)
        ratings_D['metric_D_aboveExp'] = ratings_D['metric_D_x']-ratings_D['pred_metric_D_x']
        ratings_D['metric_D_aboveAvg'] = ratings_D['metric_D_x']-metric_mean_D_D
        ratings_D['GP60_5v5'] = (ratings_D['metric_D_aboveExp']+ratings_D['metric_D_aboveAvg'])/-2
        ratings_D = ratings_D.rename(columns = {'Position_x':'Position', 'TOI_5v5_x':'TOI_5v5'})
        inseason_ratings_season_D = inseason_ratings_season_D.merge(ratings_D[['Player','PlayerID','Position','GP60_5v5']], \
            on=['Player','PlayerID','Position'])

        # D, PP
        ratings_PP_D = ratings_PP.loc[ratings_PP['Position_x']=='D']
        ratings_PP_D = ratings_PP_D.loc[ratings_PP_D['Position_y']!='G']
        ratings_PP_D = ratings_PP_D.loc[ratings_PP_D['Overlap']>0]
        ratings_PP_D = ratings_PP_D.loc[ratings_PP_D['TOI_PP_x']>0]
        ratings_PP_D = ratings_PP_D.loc[((ratings_PP_D['TOI_PK_y']>0)&(~ratings_PP_D['SameTeam']))|((ratings_PP_D['TOI_PP_y']>0)&(ratings_PP_D['SameTeam']))]
        ratings_PP_D['metricSum_PP_team'] = 0.
        ratings_PP_D.loc[ratings_PP_D['SameTeam'], 'metricSum_PP_team'] = ratings_PP_D.loc[ratings_PP_D['SameTeam'], 'metric_PP_y']*ratings_PP_D.loc[ratings_PP_D['SameTeam'], \
            'Overlap']/(ratings_PP_D.loc[ratings_PP_D['SameTeam'], 'TOI_PP_x']*4)
        ratings_PP_D['metricSum_PP_comp'] = 0.
        ratings_PP_D.loc[~ratings_PP_D['SameTeam'], 'metricSum_PP_comp'] = ratings_PP_D.loc[~ratings_PP_D['SameTeam'], 'metric_PK_y'].fillna(0)\
            *ratings_PP_D.loc[~ratings_PP_D['SameTeam'], 'Overlap']/(ratings_PP_D.loc[~ratings_PP_D['SameTeam'], 'TOI_PP_x']*4)
        ratings_PP_D = ratings_PP_D.groupby(['Player','PlayerID'], as_index=False).agg({
            'TOI_PP_x' : 'max',
            'metric_PP_x' : 'max',
            'metricSum_PP_team' : 'sum',
            'metricSum_PP_comp' : 'sum',
            'xG_PP_onice_x' : 'max',
            'Goals_PP_x' : 'max',
            'Shots_PP_x' : 'max',
            'ShotAttempts_PP_x' : 'max',
            'UnblockedShotAttempts_PP_x' : 'max',
            'xG_PP_x' : 'max',
            'xG_flurry_PP_x' : 'max',
            'PrimaryAssists_PP_x' : 'max',
            'SecondaryAssists_PP_x' : 'max',
            'GoalsAdjusted_PP_x' : 'max',
            'PrimaryAssistsAdjusted_PP_x' : 'max',
            'ShotAttemptsAdjusted_PP_x' : 'max'
        })
        ratings_PP_D.columns = ['Player','PlayerID','Position_x','TOI_PP','metric_O_PP','metricSum_team_PP','metricSum_comp_PP','xG_PP_onice','Goals_PP',
            'Shots_PP','ShotAttempts_PP','UnblockedShotAttempts_PP','xG_PP','xG_flurry_PP','PrimaryAssists_PP','SecondaryAssists_PP',
            'GoalsAdjusted_PP','PrimaryAssistsAdjusted_PP','ShotAttemptsAdjusted_PP']

        features = ['metricSum_team_PP','metricSum_comp_PP']
        X = ratings_PP_D[features].values
        Y = ratings_PP_D['metric_O_PP'].values
        model = LinearRegression()
        model.fit(X, Y)
        ratings_PP_D['pred_metric_O_PP'] = model.predict(X)
        ratings_PP_D['metric_PP_aboveExp'] = ratings_PP_D['metric_O_PP']-ratings_PP_D['pred_metric_O_PP']
        ratings_PP_D['metric_PP_aboveAvg'] = ratings_PP_D['metric_O_PP']-metric_mean_D_PP
        ratings_PP_D['indiv_contrib_PP'] = (.021*ratings_PP_D['GoalsAdjusted_PP']+.091*ratings_PP_D['PrimaryAssistsAdjusted_PP']+.033*ratings_PP_D['SecondaryAssists_PP']+\
            .354*.049399*ratings_PP_D['ShotAttempts_PP']+\
            .289*ratings_PP_D['xG_PP'])*3600/(ratings_PP_D['TOI_PP']*(.021+.091+.033+.354+.289))
        indiv_contrib_mean = (ratings_PP_D['indiv_contrib_PP']*ratings_PP_D['TOI_PP']).sum()/ratings_PP_D['TOI_PP'].sum()
        ratings_PP_D['indiv_contrib_PP_aboveAvg'] = ratings_PP_D['indiv_contrib_PP']-indiv_contrib_mean
        ratings_PP_D['GC60_PP'] = (ratings_PP_D['metric_PP_aboveExp']+ratings_PP_D['metric_PP_aboveAvg']+ratings_PP_D['indiv_contrib_PP_aboveAvg'])/3
        ratings_PP_D = ratings_PP_D.rename(columns = {'Position_x':'Position', 'TOI_PP_x':'TOI_PP'})
        inseason_ratings_season_D = inseason_ratings_season_D.merge(ratings_PP_D[['Player','PlayerID','Position','TOI_PP','GC60_PP']], \
            on=['Player','PlayerID','Position'], how='left')

        # D, PK
        ratings_PK_D = ratings_PK.loc[ratings_PK['Position_x']=='D']
        ratings_PK_D = ratings_PK_D.loc[ratings_PK_D['Position_y']!='G']
        ratings_PK_D = ratings_PK_D.loc[ratings_PK_D['Overlap']>0]
        ratings_PK_D = ratings_PK_D.loc[ratings_PK_D['TOI_PK_x']>0]
        ratings_PK_D = ratings_PK_D.loc[((ratings_PK_D['TOI_PP_y']>0)&(~ratings_PK_D['SameTeam']))|((ratings_PK_D['TOI_PK_y']>0)&(ratings_PK_D['SameTeam']))]
        ratings_PK_D['metricSum_PK_team'] = 0.
        ratings_PK_D.loc[ratings_PK_D['SameTeam'], 'metricSum_PK_team'] = ratings_PK_D.loc[ratings_PK_D['SameTeam'], 'metric_PK_y'].fillna(0)\
            *ratings_PK_D.loc[ratings_PK_D['SameTeam'], 'Overlap']/(ratings_PK_D.loc[ratings_PK_D['SameTeam'], 'TOI_PK_x']*3)
        ratings_PK_D['metricSum_PK_comp'] = 0.
        ratings_PK_D.loc[~ratings_PK_D['SameTeam'], 'metricSum_PK_comp'] = ratings_PK_D.loc[~ratings_PK_D['SameTeam'], 'metric_PP_y'].fillna(0)\
            *ratings_PK_D.loc[~ratings_PK_D['SameTeam'], 'Overlap']/(ratings_PK_D.loc[~ratings_PK_D['SameTeam'], 'TOI_PK_x']*5)
        ratings_PK_D = ratings_PK_D.groupby(['Player','PlayerID'], as_index=False).agg({
            'TOI_PK_x' : 'max',
            'metric_PK_x' : 'max',
            'metricSum_PK_team' : 'sum',
            'metricSum_PK_comp' : 'sum'
        })
        ratings_PK_D.columns = ['Player','PlayerID','TOI_PK','metric_D_PK','metricSum_team_PK','metricSum_comp_PK']

        features = ['metricSum_team_PK','metricSum_comp_PK']
        X = ratings_PK_D[features].values
        Y = ratings_PK_D['metric_D_PK'].values
        model = LinearRegression()
        model.fit(X, Y)
        ratings_PK_D['pred_metric_D_PK'] = model.predict(X)
        ratings_PK_D['metric_D_aboveExp'] = ratings_PK_D['metric_D_PK']-ratings_PK_D['pred_metric_D_PK']
        ratings_PK_D['metric_D_aboveAvg'] = ratings_PK_D['metric_D_PK']-metric_mean_D_PK
        ratings_PK_D['GP60_PK'] = (ratings_PK_D['metric_D_aboveExp']+ratings_PK_D['metric_D_aboveAvg'])/-2
        ratings_PK_D = ratings_PK_D.rename(columns = {'Position_x':'Position', 'TOI_PP_x':'TOI_PP'})
        inseason_ratings_season_D = inseason_ratings_season_D.merge(ratings_PK_D[['Player','PlayerID','Position','TOI_PK','GP60_PK']], \
            on=['Player','PlayerID','Position'], how='left')

        # D, Pens
        ratings_pen_D = xGs_temp.loc[xGs_temp['Position']=='D', ['Player','PlayerID','Position','Penalties','PenaltiesDrawn','TOI']]
        ratings_pen_D = ratings_pen_D.groupby(['Player','PlayerID','Position'], as_index=False).sum()
        pen_val = ((3600*xGs_temp['Goals_PP_onice'].sum()/xGs_temp['TOI_PP'].sum()) - (3600*xGs_temp['Goals_PK_onice'].sum()/xGs_temp['TOI_PK'].sum()))*(2/60)
        ratings_pen_D['GI60_Pens'] = pen_val*3600*(.733*ratings_pen_D['PenaltiesDrawn'].fillna(0) - 1.267*ratings_pen_D['Penalties'].fillna(0))/ratings_pen_D['TOI']
        inseason_ratings_season_D = inseason_ratings_season_D.merge(ratings_pen_D[['Player','PlayerID','Position','TOI','GI60_Pens']], \
            on=['Player','PlayerID','Position'], how='left')
        inseason_ratings_season = pd.concat([inseason_ratings_season, inseason_ratings_season_D], ignore_index=True)

        # G
        inseason_ratings_season_G = xGs_temp.loc[xGs_temp['Position']=='G', ['Player','PlayerID','Position','TOI','xGAdjustedAgainst_onice',
            'GoalsAdjustedAgainst_onice','GoalsAgainst_onice','ShotsAgainst_onice']]
        inseason_ratings_season_G = inseason_ratings_season_G.groupby(['Player','PlayerID','Position'], as_index=False).sum()
        inseason_ratings_season_G['GSAXAdjusted'] = inseason_ratings_season_G['xGAdjustedAgainst_onice'] - inseason_ratings_season_G['GoalsAdjustedAgainst_onice']
        inseason_ratings_season_G['SvPct'] = 1 - (inseason_ratings_season_G['GoalsAgainst_onice']/inseason_ratings_season_G['ShotsAgainst_onice'])
        inseason_ratings_season_G['SvPctAboveAvg'] = inseason_ratings_season_G['SvPct'] - \
            ((inseason_ratings_season_G['SvPct']*inseason_ratings_season_G['TOI']).sum()/inseason_ratings_season_G['TOI'].sum())
        inseason_ratings_season_G['GoalsSavedAboveAvg'] = inseason_ratings_season_G['SvPctAboveAvg']*inseason_ratings_season_G['ShotsAgainst_onice']
        inseason_ratings_season_G['GI60'] = (.020*inseason_ratings_season_G['GoalsSavedAboveAvg']+.018*inseason_ratings_season_G['GSAXAdjusted'])*3600/\
            (inseason_ratings_season_G['TOI']*(.020+.018))
        inseason_ratings_season_G = inseason_ratings_season_G[['Player','PlayerID','Position','TOI','GI60']]
        inseason_ratings_season = inseason_ratings_season.merge(inseason_ratings_season_G, on=['Player','PlayerID','Position','TOI'], how='outer')

        # combine
        inseason_ratings_season['Date'] = i
        inseason_ratings_season['Season'] = season
        inseason_ratings_season.columns = ['Player','PlayerID','Position','prevGames_TOI_5v5','prevGames_GC60_5v5','prevGames_GP60_5v5',
            'prevGames_TOI_PP','prevGames_GC60_PP','prevGames_TOI_PK','prevGames_GP60_PK','prevGames_TOI','prevGames_GI60_Pens','prevGames_GI60',
            'PlayerGameNum','Season']
        if len(inseason_ratings.index)==0:
            inseason_ratings = inseason_ratings_season.copy(deep=True)
        else:
            inseason_ratings = pd.concat([inseason_ratings, inseason_ratings_season], ignore_index=True)

    # merge with original data
    playerGame = playerGame.merge(inseason_ratings, on=['Player','PlayerID','Position','Season','Date'], how='left')
    return playerGame


def add_team_inseason_ratings(teamGame):
    # add stats from previous games in the season to each row of teamGame
    metrics = ['Goals','Shots','ShotAttempts','UnblockedShotAttempts','xG','xG_flurry',
        'Goals_5v5','Shots_5v5','ShotAttempts_5v5','UnblockedShotAttempts_5v5','xG_5v5','xG_flurry_5v5',
        'GoalsAdjusted','ShotsAdjusted','ShotAttemptsAdjusted',
            'UnblockedShotAttemptsAdjusted','xGAdjusted','xG_flurryAdjusted',
        'GoalsAdjusted_5v5','ShotsAdjusted_5v5','ShotAttemptsAdjusted_5v5',
            'UnblockedShotAttemptsAdjusted_5v5','xGAdjusted_5v5','xG_flurryAdjusted_5v5',
        'Opp_Goals','Opp_Shots','Opp_ShotAttempts',
            'Opp_UnblockedShotAttempts','Opp_xG','Opp_xG_flurry',
        'Opp_Goals_5v5','Opp_Shots_5v5','Opp_ShotAttempts_5v5',
            'Opp_UnblockedShotAttempts_5v5','Opp_xG_5v5','Opp_xG_flurry_5v5',
        'Opp_GoalsAdjusted','Opp_ShotsAdjusted','Opp_ShotAttemptsAdjusted',
            'Opp_UnblockedShotAttemptsAdjusted','Opp_xGAdjusted','Opp_xG_flurryAdjusted',
        'Opp_GoalsAdjusted_5v5','Opp_ShotsAdjusted_5v5','Opp_ShotAttemptsAdjusted_5v5',
            'Opp_UnblockedShotAttemptsAdjusted_5v5','Opp_xGAdjusted_5v5','Opp_xG_flurryAdjusted_5v5']
    inseason_ratings = pd.DataFrame()
    for i in teamGame['Date'].sort_values().unique()[1:]:
        curSeason = teamGame.loc[teamGame['Date']==i, 'Season'].max()
        df_temp = teamGame.loc[(teamGame['Date']<i)&(teamGame['Season']==curSeason)]
        df_temp = df_temp.loc[df_temp['Team'].isin(teamGame.loc[teamGame['Date']==i, 'Team'])]
        if len(df_temp.index)>0:
            df_temp = df_temp[['Team','Season']+metrics].groupby(['Team','Season'], as_index=False).mean()
            df_temp.columns = ['Team','Season',
                'prevGames_Goals','prevGames_Shots','prevGames_ShotAttempts','prevGames_UnblockedShotAttempts','prevGames_xG','prevGames_xG_flurry',
                'prevGames_Goals_5v5','prevGames_Shots_5v5','prevGames_ShotAttempts_5v5','prevGames_UnblockedShotAttempts_5v5','prevGames_xG_5v5','prevGames_xG_flurry_5v5',
                'prevGames_GoalsAdjusted','prevGames_ShotsAdjusted','prevGames_ShotAttemptsAdjusted',
                    'prevGames_UnblockedShotAttemptsAdjusted','prevGames_xGAdjusted','prevGames_xG_flurryAdjusted',
                'prevGames_GoalsAdjusted_5v5','prevGames_ShotsAdjusted_5v5','prevGames_ShotAttemptsAdjusted_5v5',
                    'prevGames_UnblockedShotAttemptsAdjusted_5v5','prevGames_xGAdjusted_5v5','prevGames_xG_flurryAdjusted_5v5',
                'prevGames_GoalsAgainst','prevGames_ShotsAgainst','prevGames_ShotAttemptsAgainst',
                    'prevGames_UnblockedShotAttemptsAgainst','prevGames_xGAgainst','prevGames_xG_flurryAgainst',
                'prevGames_Goals_5v5Against','prevGames_Shots_5v5Against','prevGames_ShotAttempts_5v5Against',
                    'prevGames_UnblockedShotAttempts_5v5Against','prevGames_xG_5v5Against','prevGames_xG_flurry_5v5Against',
                'prevGames_GoalsAdjustedAgainst','prevGames_ShotsAdjustedAgainst','prevGames_ShotAttemptsAdjustedAgainst',
                    'prevGames_UnblockedShotAttemptsAdjustedAgainst','prevGames_xGAdjustedAgainst','prevGames_xG_flurryAdjustedAgainst',
                'prevGames_GoalsAdjusted_5v5Against','prevGames_ShotsAdjusted_5v5Against','prevGames_ShotAttemptsAdjusted_5v5Against',
                    'prevGames_UnblockedShotAttemptsAdjusted_5v5Against','prevGames_xGAdjusted_5v5Against','prevGames_xG_flurryAdjusted_5v5Against']
            df_temp['Date'] = i
            if len(inseason_ratings.index)==0:
                inseason_ratings = df_temp.copy(deep=True)
            else:
                inseason_ratings = pd.concat([inseason_ratings, df_temp], ignore_index=True)


    teamGame = teamGame.merge(inseason_ratings, on=['Team','Season','Date'], how='left')
    return teamGame
