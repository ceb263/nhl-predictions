import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

player_config_file = 'configs/preseason_config_player.json'
with open(player_config_file, 'r') as f:
    preseason_config_player = json.load(f)

team_config_file = 'configs/preseason_config_team.json'
with open(team_config_file, 'r') as f:
    preseason_config_team = json.load(f)

class PreseasonRatingsUpdater(object):
    def __init__(self, player_file='data/ratings_preseason_players.csv', team_file='data/ratings_preseason_teams.csv'):
        self.player_file = player_file
        self.model_file = team_file

    def _predict_players(self, ratings, metric, toi_col, weights, season):
        # used within _predict_players_season
        df_1 = ratings.loc[ratings['Season']==season-1]
        df_1 = df_1[['Player','PlayerID','Position',toi_col,metric]]
        df_1['feature1'] = df_1[metric]*df_1[toi_col]/50000.
        df_2 = ratings.loc[ratings['Season']==season-2]
        df_2 = df_2[['Player','PlayerID','Position',toi_col,metric]]
        df_2['feature2'] = df_2[metric]*df_2[toi_col]/50000.
        df_3 = ratings.loc[ratings['Season']==season-3]
        df_3 = df_3[['Player','PlayerID','Position',toi_col,metric]]
        df_3['feature3'] = df_3[metric]*df_3[toi_col]/50000.

        df_1 = df_1.merge(df_2, on=['Player','PlayerID','Position'], how='left')
        df_1 = df_1.merge(df_3, on=['Player','PlayerID','Position'], how='left')

        df_1.loc[df_1['feature2'].isna(), 'x'+metric] = \
            weights[1][0] + df_1.loc[df_1['feature2'].isna(), 'feature1']*weights[1][1]
        df_1.loc[(~df_1['feature2'].isna())&(df_1['feature3'].isna()), 'x'+metric] = \
            weights[2][0] + df_1.loc[(~df_1['feature2'].isna())&(df_1['feature3'].isna()), 'feature1']*weights[2][1] \
            + df_1.loc[(~df_1['feature2'].isna())&(df_1['feature3'].isna()), 'feature2']*weights[2][2]
        df_1.loc[(~df_1['feature2'].isna())&(~df_1['feature3'].isna()), 'x'+metric] = \
            weights[3][0] + df_1.loc[(~df_1['feature2'].isna())&(~df_1['feature3'].isna()), 'feature1']*weights[3][1] \
            + df_1.loc[(~df_1['feature2'].isna())&(~df_1['feature3'].isna()), 'feature2']*weights[3][2] \
            + df_1.loc[(~df_1['feature2'].isna())&(~df_1['feature3'].isna()), 'feature3']*weights[3][3]

        return df_1[['Player','PlayerID','Position','x'+metric]]

    def _predict_players_season(self, ratings, season):
        # used within update_player_preseason_ratings()
        df = ratings.loc[ratings['Season']==season-1]
        df = df[['Player','PlayerID','Position']]
        fix_metrics = []
        for pos, val in preseason_config_player.items():
            ratings_temp = ratings.loc[ratings['Position']==pos]
            if pos=='G':
                df = df.merge(self._predict_players(ratings_temp, 'GI60', 'TOI', val, season), on=['Player','PlayerID','Position'], how='left')
            else:
                for strength, val2 in val.items():
                    for metric, val3 in val2.items():
                        if strength=='Pens':
                            toi_col = 'TOI'
                        else:
                            toi_col = 'TOI_'+strength
                        df = df.merge(self._predict_players(ratings_temp, metric+'_'+strength, toi_col, val3, season), on=['Player','PlayerID','Position'], how='left')
                        if 'x'+metric+'_'+strength not in fix_metrics:
                            fix_metrics.append('x'+metric+'_'+strength)

        for m in fix_metrics:
            df[m] = df[[m+'_x', m+'_y']].max(axis=1)
            df = df.drop(columns=[m+'_x', m+'_y'])

        return df

    def update_player_preseason_ratings(self, start_season=2015, end_season=2022, out_file='data/ratings_preseason_players.csv'):
        ratings_all = pd.DataFrame()
        for season in range(start_season-3, end_season):
            # read data
            playerGame = pd.read_pickle('data/playerGame_{}.pkl'.format(str(season)))
            toi_overlap = pd.read_pickle('data/toiOverlap_{}.pkl'.format(str(season)))

            xGs = playerGame.loc[playerGame['Playoffs']==0]
            xGs = xGs.drop(columns=['Playoffs','DateInt','Season','Date','Game_Id','Playoffs','DateInt','PlayerGameID','Team'])
            xGs = xGs.groupby(['Player','PlayerID','Position'], as_index=False).sum()
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

            #compute mean metrics
            metric_mean_F_O = 3600*(
                .163*xGs.loc[xGs['Position']=='F', 'Goals_5v5_onice'].sum()
                + .185*.091286*xGs.loc[xGs['Position']=='F', 'ShotsAdjusted_5v5_onice'].sum()
                + .262*xGs.loc[xGs['Position']=='F', 'xG_flurry_5v5_onice'].sum()
                )/(xGs.loc[xGs['Position']=='F', 'TOI_5v5'].sum() * (.163+.185+.262))
            metric_mean_D_O = 3600*(
                .064*xGs.loc[xGs['Position']=='D', 'GoalsAdjusted_5v5_onice'].sum()
                + .132*.049399*xGs.loc[xGs['Position']=='D', 'ShotAttemptsAdjusted_5v5_onice'].sum()
                + .160*xGs.loc[xGs['Position']=='D', 'xG_5v5_onice'].sum()
                )/(xGs.loc[xGs['Position']=='D', 'TOI_5v5'].sum() * (.064+.132+.160))
            metric_mean_F_D = 3600*(
                .031*xGs.loc[xGs['Position']=='F', 'GoalsAgainst_5v5_onice'].sum()
                + .140*.065956*xGs.loc[xGs['Position']=='F', 'UnblockedShotAttemptsAdjustedAgainst_5v5_onice'].sum()
                + .175*xGs.loc[xGs['Position']=='F', 'xG_flurryAdjustedAgainst_5v5_onice'].sum()
                )/(xGs.loc[xGs['Position']=='F', 'TOI_5v5'].sum() * (.031+.140+.175))
            metric_mean_D_D = 3600*(
                .041*xGs.loc[xGs['Position']=='D', 'GoalsAgainst_5v5_onice'].sum()
                + .172*.065956*xGs.loc[xGs['Position']=='D', 'UnblockedShotAttemptsAdjustedAgainst_5v5_onice'].sum()
                + .216*xGs.loc[xGs['Position']=='D', 'xGAdjustedAgainst_5v5_onice'].sum()
                )/(xGs.loc[xGs['Position']=='D', 'TOI_5v5'].sum() * (.041+.172+.216))
            metric_mean_F_PP = 3600*(
                .154*xGs.loc[xGs['Position']=='F', 'GoalsAdjusted_PP_onice'].sum()
                + .268*.065956*xGs.loc[xGs['Position']=='F', 'UnblockedShotAttempts_PP_onice'].sum()
                + .293*xGs.loc[xGs['Position']=='F', 'xG_flurry_PP_onice'].sum()
                )/(xGs.loc[xGs['Position']=='F', 'TOI_PP'].sum() * (.154+.268+.293))
            metric_mean_D_PP = 3600*(
                .104*xGs.loc[xGs['Position']=='D', 'GoalsAdjusted_PP_onice'].sum()
                + .249*.065956*xGs.loc[xGs['Position']=='D', 'UnblockedShotAttempts_PP_onice'].sum()
                + .215*xGs.loc[xGs['Position']=='D', 'xG_flurry_PP_onice'].sum()
                )/(xGs.loc[xGs['Position']=='D', 'TOI_PP'].sum() * (.104+.249+.215))
            metric_mean_F_PK = 3600*(
                .019*xGs.loc[xGs['Position']=='F', 'GoalsAdjustedAgainst_PK_onice'].sum()
                + .155*.049399*xGs.loc[xGs['Position']=='F', 'ShotAttemptsAgainst_PK_onice'].sum()
                + .097*xGs.loc[xGs['Position']=='F', 'xG_flurryAdjustedAgainst_PK_onice'].sum()
                )/(xGs.loc[xGs['Position']=='F', 'TOI_PK'].sum() * (.019+.155+.097))
            metric_mean_D_PK = 3600*(
                .017*xGs.loc[xGs['Position']=='D', 'GoalsAgainst_PK_onice'].sum()
                + .166*.049399*xGs.loc[xGs['Position']=='D', 'ShotAttemptsAgainst_PK_onice'].sum()
                + .065*xGs.loc[xGs['Position']=='D', 'xG_flurryAgainst_PK_onice'].sum()
                )/(xGs.loc[xGs['Position']=='D', 'TOI_PK'].sum() * (.017+.166+.065))

            ratings = toi_overlap.rename(columns={'Player_x':'Player', 'Player_Id_x':'PlayerID'})
            ratings = ratings.loc[ratings['Strength']=='5x5']
            ratings = ratings[['Player','PlayerID','Player_y','Player_Id_y','SameTeam','Overlap']]
            ratings = ratings.groupby(['Player','PlayerID','Player_y','Player_Id_y','SameTeam'], as_index=False).sum()
            ratings = ratings.merge(xGs, on=['Player','PlayerID'])
            xG_others = xGs.rename(columns={'Player':'Player_y', 'PlayerID':'Player_Id_y'})
            ratings = ratings.merge(xG_others, on=['Player_y','Player_Id_y'])
            del xG_others

            ratings_PP = toi_overlap.rename(columns={'Player_x':'Player', 'Player_Id_x':'PlayerID'})
            ratings_PP = ratings_PP.loc[ratings_PP['Strength'].isin(['5x4','5x3','4x3'])]
            ratings_PP = ratings_PP[['Player','PlayerID','Player_y','Player_Id_y','SameTeam','Overlap']]
            ratings_PP = ratings_PP.groupby(['Player','PlayerID','Player_y','Player_Id_y','SameTeam'], as_index=False).sum()
            ratings_PP = ratings_PP.merge(xGs, on=['Player','PlayerID'])
            xG_others = xGs.rename(columns={'Player':'Player_y', 'PlayerID':'Player_Id_y'})
            ratings_PP = ratings_PP.merge(xG_others, on=['Player_y','Player_Id_y'])
            del xG_others

            ratings_PK = toi_overlap.rename(columns={'Player_x':'Player', 'Player_Id_x':'PlayerID'})
            ratings_PK = ratings_PK.loc[ratings_PK['Strength'].isin(['4x5','3x5','3x4'])]
            ratings_PK = ratings_PK[['Player','PlayerID','Player_y','Player_Id_y','SameTeam','Overlap']]
            ratings_PK = ratings_PK.groupby(['Player','PlayerID','Player_y','Player_Id_y','SameTeam'], as_index=False).sum()
            ratings_PK = ratings_PK.merge(xGs, on=['Player','PlayerID'])
            xG_others = xGs.rename(columns={'Player':'Player_y', 'PlayerID':'Player_Id_y'})
            ratings_PK = ratings_PK.merge(xG_others, on=['Player_y','Player_Id_y'])
            del xG_others

            ratings_F = ratings.loc[ratings['Position_x']=='F']
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
            ratings_F['adjustment'] = 1+5*((np.log(ratings_F['TOI_5v5_x'])/np.log(ratings_F['TOI_5v5_x'].mean()))-1)
            ratings_F['nonneg_val'] = 0.0
            ratings_F['nonneg_adjustment'] = ratings_F[['adjustment','nonneg_val']].max(axis=1)
            ratings_F['GC60_5v5_Adj'] = ratings_F['GC60_5v5']*ratings_F['nonneg_adjustment']
            ratings_F['GC_5v5'] = ratings_F['GC60_5v5']*ratings_F['TOI_5v5_x']/3600

            features = ['DZoneStartRate_5v5_x','metricSum_D_team','metricSum_D_comp']
            X = ratings_F.dropna(subset=features)[features].values
            Y = ratings_F.dropna(subset=features)['metric_D_x'].values

            model = LinearRegression()
            model.fit(X, Y)
            ratings_F['pred_metric_D_x'] = model.predict(ratings_F[features].fillna(0.4).values)
            ratings_F['metric_D_aboveExp'] = ratings_F['metric_D_x']-ratings_F['pred_metric_D_x']
            ratings_F['metric_D_aboveAvg'] = ratings_F['metric_D_x']-metric_mean_F_D
            ratings_F['GP60_5v5'] = (ratings_F['metric_D_aboveExp']+ratings_F['metric_D_aboveAvg'])/-2
            ratings_F['adjustment'] = 1+5*((np.log(ratings_F['TOI_5v5_x'])/np.log(ratings_F['TOI_5v5_x'].mean()))-1)
            ratings_F['nonneg_val'] = 0.0
            ratings_F['nonneg_adjustment'] = ratings_F[['adjustment','nonneg_val']].max(axis=1)
            ratings_F['GP60_5v5_Adj'] = ratings_F['GP60_5v5']*ratings_F['nonneg_adjustment']
            ratings_F['GP_5v5'] = ratings_F['GP60_5v5']*ratings_F['TOI_5v5_x']/3600

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
            ratings_PP_F.columns = ['Player','PlayerID','TOI_PP','metric_O_PP','metricSum_team_PP','metricSum_comp_PP','xG_PP_onice','Goals_PP',
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
            ratings_PP_F['adjustment'] = 1+5*((np.log(ratings_PP_F['TOI_PP'])/np.log(ratings_PP_F['TOI_PP'].mean()))-1)
            ratings_PP_F['nonneg_val'] = 0.0
            ratings_PP_F['nonneg_adjustment'] = ratings_PP_F[['adjustment','nonneg_val']].max(axis=1)
            ratings_PP_F['GC60_PP_Adj'] = ratings_PP_F['GC60_PP']*ratings_PP_F['nonneg_adjustment']
            ratings_PP_F['GC_PP'] = ratings_PP_F['GC60_PP']*ratings_PP_F['TOI_PP']/3600

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
            ratings_PK_F.columns = ['Player','PlayerID','TOI_PK','metric_D_PK','metricSum_team_PK','metricSum_comp_PK']

            features = ['metricSum_team_PK','metricSum_comp_PK']
            X = ratings_PK_F.loc[ratings_PK_F['TOI_PK']>100, features].values
            Y = ratings_PK_F.loc[ratings_PK_F['TOI_PK']>100, 'metric_D_PK'].values

            model = LinearRegression()
            model.fit(X, Y)
            ratings_PK_F['pred_metric_D_PK'] = model.predict(ratings_PK_F[features].values)
            ratings_PK_F['metric_D_aboveExp'] = ratings_PK_F['metric_D_PK']-ratings_PK_F['pred_metric_D_PK']
            ratings_PK_F['metric_D_aboveAvg'] = ratings_PK_F['metric_D_PK']-metric_mean_F_PK
            ratings_PK_F['GP60_PK'] = (ratings_PK_F['metric_D_aboveExp']+ratings_PK_F['metric_D_aboveAvg'])/-2
            ratings_PK_F['adjustment'] = 1+5*((np.log(ratings_PK_F['TOI_PK'])/np.log(ratings_PK_F['TOI_PK'].mean()))-1)
            ratings_PK_F['nonneg_val'] = 0.0
            ratings_PK_F['nonneg_adjustment'] = ratings_PK_F[['adjustment','nonneg_val']].max(axis=1)
            ratings_PK_F['GP60_PK_Adj'] = ratings_PK_F['GP60_PK']*ratings_PK_F['nonneg_adjustment']
            ratings_PK_F['GP_PK'] = ratings_PK_F['GP60_PK']*ratings_PK_F['TOI_PK']/3600

            ratings_pen_F = xGs.loc[xGs['Position']=='F', ['Player','PlayerID','Penalties','PenaltiesDrawn','TOI']]
            ratings_pen_F = ratings_pen_F.groupby(['Player','PlayerID'], as_index=False).sum()

            # get value (in goals) of a penalty, and then use that to calculate goal impact from taking/drawing penalties
            pen_val = ((3600*xGs['Goals_PP_onice'].sum()/xGs['TOI_PP'].sum()) - (3600*xGs['Goals_PK_onice'].sum()/xGs['TOI_PK'].sum()))*(2/60)
            ratings_pen_F['GI60_Pens'] = pen_val*3600*(.837*ratings_pen_F['PenaltiesDrawn'].fillna(0) - 1.163*ratings_pen_F['Penalties'].fillna(0))/ratings_pen_F['TOI']
            ratings_pen_F['adjustment'] = 1+5*((np.log(ratings_pen_F['TOI'])/np.log(ratings_pen_F['TOI'].mean()))-1)
            ratings_pen_F['nonneg_val'] = 0.0
            ratings_pen_F['nonneg_adjustment'] = ratings_pen_F[['adjustment','nonneg_val']].max(axis=1)
            ratings_pen_F['GI60_Pens_Adj'] = ratings_pen_F['GI60_Pens']*ratings_pen_F['nonneg_adjustment']
            ratings_pen_F['GI_Pens'] = ratings_pen_F['GI60_Pens']*ratings_pen_F['TOI']/3600

            ratings_PP_F = ratings_PP_F[['Player','PlayerID','TOI_PP','GC60_PP']]
            ratings_PK_F = ratings_PK_F[['Player','PlayerID','TOI_PK','GP60_PK']]
            ratings_pen_F = ratings_pen_F[['Player','PlayerID','GI60_Pens']]
            ratings_F = ratings_F[['Player','PlayerID','Position_x','TOI_x','TOI_5v5_x','GC60_5v5','GP60_5v5']]
            ratings_F = ratings_F.merge(ratings_PP_F, on=['Player','PlayerID'], how='left')
            ratings_F = ratings_F.merge(ratings_PK_F, on=['Player','PlayerID'], how='left')
            ratings_F = ratings_F.merge(ratings_pen_F, on=['Player','PlayerID'], how='left').fillna(0.)

            ratings_F['GI60'] = ratings_F['GI60_Pens'] + \
                ((ratings_F['TOI_5v5_x']*(ratings_F['GC60_5v5']+ratings_F['GP60_5v5']) + ratings_F['TOI_PP']*ratings_F['GC60_PP']\
                + ratings_F['TOI_PK']*ratings_F['GP60_PK']) / (ratings_F['TOI_5v5_x']+ratings_F['TOI_PP']+ratings_F['TOI_PK']))
            ratings_F['GI'] = ratings_F['GI60']*ratings_F['TOI_x']/3600

            ratings_D = ratings.loc[ratings['Position_x']=='D']
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
            ratings_D['adjustment'] = 1+5*((np.log(ratings_D['TOI_5v5_x'])/np.log(ratings_D['TOI_5v5_x'].mean()))-1)
            ratings_D['nonneg_val'] = 0.0
            ratings_D['nonneg_adjustment'] = ratings_D[['adjustment','nonneg_val']].max(axis=1)
            ratings_D['GC60_5v5_Adj'] = ratings_D['GC60_5v5']*ratings_D['nonneg_adjustment']
            ratings_D['GC_5v5'] = ratings_D['GC60_5v5']*ratings_D['TOI_5v5_x']/3600

            features = ['DZoneStartRate_5v5_x','metricSum_D_team','metricSum_D_comp']
            X = ratings_D.dropna(subset=features)[features].values
            Y = ratings_D.dropna(subset=features)['metric_D_x'].values

            model = LinearRegression()
            model.fit(X, Y)
            ratings_D['pred_metric_D_x'] = model.predict(ratings_D[features].fillna(0.4).values)
            ratings_D['metric_D_aboveExp'] = ratings_D['metric_D_x']-ratings_D['pred_metric_D_x']
            ratings_D['metric_D_aboveAvg'] = ratings_D['metric_D_x']-metric_mean_D_D
            ratings_D['GP60_5v5'] = (ratings_D['metric_D_aboveExp']+ratings_D['metric_D_aboveAvg'])/-2
            ratings_D['adjustment'] = 1+5*((np.log(ratings_D['TOI_5v5_x'])/np.log(ratings_D['TOI_5v5_x'].mean()))-1)
            ratings_D['nonneg_val'] = 0.0
            ratings_D['nonneg_adjustment'] = ratings_D[['adjustment','nonneg_val']].max(axis=1)
            ratings_D['GP60_5v5_Adj'] = ratings_D['GP60_5v5']*ratings_D['nonneg_adjustment']
            ratings_D['GP_5v5'] = ratings_D['GP60_5v5']*ratings_D['TOI_5v5_x']/3600

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
                'GoalsAdjusted_PP_x' : 'max',
                'Shots_PP_x' : 'max',
                'ShotAttempts_PP_x' : 'max',
                'ShotAttemptsAdjusted_PP_x' : 'max',
                'UnblockedShotAttempts_PP_x' : 'max',
                'xG_PP_x' : 'max',
                'xG_flurry_PP_x' : 'max',
                'PrimaryAssists_PP_x' : 'max',
                'PrimaryAssistsAdjusted_PP_x' : 'max',
                'SecondaryAssists_PP_x' : 'max'
            })
            ratings_PP_D.columns = ['Player','PlayerID','TOI_PP','metric_O_PP','metricSum_team_PP','metricSum_comp_PP','xG_PP_onice','Goals_PP','GoalsAdjusted_PP',
                'Shots_PP','ShotAttempts_PP','ShotAttemptsAdjusted_PP','UnblockedShotAttempts_PP','xG_PP','xG_flurry_PP','PrimaryAssists_PP','SecondaryAssists_PP',
                'PrimaryAssistsAdjusted_PP']

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
            ratings_PP_D['adjustment'] = 1+5*((np.log(ratings_PP_D['TOI_PP'])/np.log(ratings_PP_D['TOI_PP'].mean()))-1)
            ratings_PP_D['nonneg_val'] = 0.0
            ratings_PP_D['nonneg_adjustment'] = ratings_PP_D[['adjustment','nonneg_val']].max(axis=1)
            ratings_PP_D['GC60_PP_Adj'] = ratings_PP_D['GC60_PP']*ratings_PP_D['nonneg_adjustment']
            ratings_PP_D['GC_PP'] = ratings_PP_D['GC60_PP']*ratings_PP_D['TOI_PP']/3600

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
            ratings_PK_D['adjustment'] = 1+5*((np.log(ratings_PK_D['TOI_PK'])/np.log(ratings_PK_D['TOI_PK'].mean()))-1)
            ratings_PK_D['nonneg_val'] = 0.0
            ratings_PK_D['nonneg_adjustment'] = ratings_PK_D[['adjustment','nonneg_val']].max(axis=1)
            ratings_PK_D['GP60_PK_Adj'] = ratings_PK_D['GP60_PK']*ratings_PK_D['nonneg_adjustment']
            ratings_PK_D['GP_PK'] = ratings_PK_D['GP60_PK']*ratings_PK_D['TOI_PK']/3600

            ratings_pen_D = xGs.loc[xGs['Position']=='D', ['Player','PlayerID','Penalties','PenaltiesDrawn','TOI']]
            ratings_pen_D = ratings_pen_D.groupby(['Player','PlayerID'], as_index=False).sum()

            # get value (in goals) of a penalty, and then use that to calculate goal impact from taking/drawing penalties
            pen_val = ((3600*xGs['Goals_PP_onice'].sum()/xGs['TOI_PP'].sum()) - (3600*xGs['Goals_PK_onice'].sum()/xGs['TOI_PK'].sum()))*(2/60)
            ratings_pen_D['GI60_Pens'] = pen_val*3600*(.733*ratings_pen_D['PenaltiesDrawn'].fillna(0) - 1.267*ratings_pen_D['Penalties'].fillna(0))/ratings_pen_D['TOI']
            ratings_pen_D['adjustment'] = 1+5*((np.log(ratings_pen_D['TOI'])/np.log(ratings_pen_D['TOI'].mean()))-1)
            ratings_pen_D['nonneg_val'] = 0.0
            ratings_pen_D['nonneg_adjustment'] = ratings_pen_D[['adjustment','nonneg_val']].max(axis=1)
            ratings_pen_D['GI60_Pens_Adj'] = ratings_pen_D['GI60_Pens']*ratings_pen_D['nonneg_adjustment']
            ratings_pen_D['GI_Pens'] = ratings_pen_D['GI60_Pens']*ratings_pen_D['TOI']/3600

            ratings_PP_D = ratings_PP_D[['Player','PlayerID','TOI_PP','GC60_PP']]
            ratings_PK_D = ratings_PK_D[['Player','PlayerID','TOI_PK','GP60_PK']]
            ratings_pen_D = ratings_pen_D[['Player','PlayerID','GI60_Pens']]
            ratings_D = ratings_D[['Player','PlayerID','Position_x','TOI_x','TOI_5v5_x','GC60_5v5','GP60_5v5']]
            ratings_D = ratings_D.merge(ratings_PP_D, on=['Player','PlayerID'], how='left')
            ratings_D = ratings_D.merge(ratings_PK_D, on=['Player','PlayerID'], how='left')
            ratings_D = ratings_D.merge(ratings_pen_D, on=['Player','PlayerID'], how='left').fillna(0.)

            ratings_D['GI60'] = ratings_D['GI60_Pens'] + \
                ((ratings_D['TOI_5v5_x']*(ratings_D['GC60_5v5']+ratings_D['GP60_5v5']) + ratings_D['TOI_PP']*ratings_D['GC60_PP']\
                + ratings_D['TOI_PK']*ratings_D['GP60_PK']) / (ratings_D['TOI_5v5_x']+ratings_D['TOI_PP']+ratings_D['TOI_PK']))
            ratings_D['GI'] = ratings_D['GI60']*ratings_D['TOI_x']/3600

            ratings_G = xGs.loc[xGs['Position']=='G', ['Player','PlayerID','ShotsAgainst_onice','GoalsAgainst_onice','TOI','xGAgainst_onice','xG_flurryAgainst_onice',
                'ShotsAdjustedAgainst_onice','GoalsAdjustedAgainst_onice','xGAdjustedAgainst_onice','xG_flurryAdjustedAgainst_onice','ReboundShotsAgainst_onice']]
            ratings_G = ratings_G.groupby(['Player','PlayerID'], as_index=False).sum()

            ratings_G['GSAXAdjusted'] = ratings_G['xGAdjustedAgainst_onice'] - ratings_G['GoalsAdjustedAgainst_onice']
            ratings_G['SvPct'] = 1 - (ratings_G['GoalsAgainst_onice']/ratings_G['ShotsAgainst_onice'])
            ratings_G['SvPctAboveAvg'] = ratings_G['SvPct'] - ((ratings_G['SvPct']*ratings_G['TOI']).sum()/ratings_G['TOI'].sum())
            ratings_G['GoalsSavedAboveAvg'] = ratings_G['SvPctAboveAvg']*ratings_G['ShotsAgainst_onice']
            ratings_G['GI60'] = (.020*ratings_G['GoalsSavedAboveAvg']+.018*ratings_G['GSAXAdjusted'])*3600/\
                (ratings_G['TOI']*(.020+.018))

            ratings_G['adjustment'] = 1+10*((np.log(ratings_G['TOI'])/np.log(ratings_G['TOI'].mean()))-1)
            ratings_G['nonneg_val'] = 0.0
            ratings_G['nonneg_adjustment'] = ratings_G[['adjustment','nonneg_val']].max(axis=1)
            ratings_G['GP60_Adj'] = ratings_G['GI60']*ratings_G['nonneg_adjustment']
            ratings_G['GI'] = ratings_G['GI60']*ratings_G['TOI']/3600

            #combine and output rankings
            combined_cols = ['Player','PlayerID','Position_x','TOI_x','GI60','TOI_5v5_x','GC60_5v5','GP60_5v5','TOI_PP','GC60_PP','TOI_PK','GP60_PK','GI60_Pens']
            ratings_G['TOI_x'] = ratings_G['TOI']
            ratings_G['Position_x'] = 'G'
            ratings_out = pd.concat([ratings_F[combined_cols], ratings_D[combined_cols], ratings_G[['Player','PlayerID','Position_x','TOI_x','GI60']]], ignore_index=True)
            ratings_out.columns = ['Player','PlayerID','Position','TOI','GI60','TOI_5v5','GC60_5v5','GP60_5v5','TOI_PP','GC60_PP','TOI_PK','GP60_PK','GI60_Pens']
            ratings_out['Season'] = season
            ratings_all = pd.concat([ratings_all, ratings_out], ignore_index=True)

        preseason_preds = pd.DataFrame()
        for season in range(start_season, end_season+1):
            preds_season = self._predict_players_season(ratings_all, season)
            preds_season['Season'] = season
            preseason_preds = pd.concat([preseason_preds, preds_season], ignore_index=True)

        preseason_preds.to_csv(out_file, index=False)

    def _predict_teams(self, ratings, metric, weights, season):
        df_2 = ratings.loc[ratings['Season']==season-1]
        df_2 = df_2[['Team',metric]]
        df_1 = ratings.loc[ratings['Season']==season]
        df_1 = df_1[['Team']]

        df = df_2.merge(df_1, on=['Team'], how='outer')
        df.loc[df[metric].isna(), 'x'+metric] = weights[0][0]
        df.loc[~df[metric].isna(), 'x'+metric] = \
            weights[1][0] + df.loc[~df[metric].isna(), metric]*weights[1][1]

        return df[['Team','x'+metric]]

    def _predict_teams_season(self, ratings, season):
        df = ratings[['Team']].drop_duplicates()
        for metric, val in preseason_config_team.items():
            df = df.merge(self._predict_teams(ratings[['Team','Season',metric]], metric, val, season), on=['Team'], how='left')

        return df

    def update_team_preseason_ratings(self, start_season=2015, end_season=2022, out_file='data/ratings_preseason_teams.csv'):
        teamGame = pd.read_pickle('data/teamGame.pkl')
        teamSeason = teamGame.loc[teamGame['Playoffs']==0, ['Game_Id','Date','Team','Season',
            'Goals','Shots','ShotAttempts','UnblockedShotAttempts','xG','xG_flurry','Goals_5v5','Shots_5v5','ShotAttempts_5v5',
            'UnblockedShotAttempts_5v5','xG_5v5','xG_flurry_5v5',
            'GoalsAdjusted','ShotsAdjusted','ShotAttemptsAdjusted','UnblockedShotAttemptsAdjusted','xGAdjusted','xG_flurryAdjusted',
            'GoalsAdjusted_5v5','ShotsAdjusted_5v5','ShotAttemptsAdjusted_5v5','UnblockedShotAttemptsAdjusted_5v5',
            'xGAdjusted_5v5','xG_flurryAdjusted_5v5']]
        del teamGame
        teamSeason_Opp = teamSeason.copy(deep=True)
        teamSeason_Opp.columns = ['Game_Id','Date','Opp','Season',
            'GoalsAgainst','ShotsAgainst','ShotAttemptsAgainst','UnblockedShotAttemptsAgainst','xGAgainst','xG_flurryAgainst',
            'Goals_5v5Against','Shots_5v5Against','ShotAttempts_5v5Against','UnblockedShotAttempts_5v5Against','xG_5v5Against','xG_flurry_5v5Against',
            'GoalsAdjustedAgainst','ShotsAdjustedAgainst','ShotAttemptsAdjustedAgainst','UnblockedShotAttemptsAdjustedAgainst','xGAdjustedAgainst','xG_flurryAdjustedAgainst',
            'GoalsAdjusted_5v5Against','ShotsAdjusted_5v5Against','ShotAttemptsAdjusted_5v5Against','UnblockedShotAttemptsAdjusted_5v5Against',
            'xGAdjusted_5v5Against','xG_flurryAdjusted_5v5Against']
        teamSeason = teamSeason.merge(teamSeason_Opp, on=['Game_Id','Date','Season'])
        del teamSeason_Opp
        teamSeason = teamSeason.loc[teamSeason['Team']!=teamSeason['Opp']]
        teamSeason = teamSeason.drop(columns=['Game_Id','Date','Opp']).groupby(['Team','Season'], as_index=False).mean()

        preseason_preds = pd.DataFrame()
        for season in range(start_season, end_season+1):
            preds_season = self._predict_teams_season(teamSeason, season)
            preds_season['Season'] = season
            preseason_preds = pd.concat([preseason_preds, preds_season], ignore_index=True)

        preseason_preds.to_csv(out_file, index=False)
