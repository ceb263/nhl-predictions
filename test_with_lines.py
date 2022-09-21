import data_processing
import inseason_ratings
import games_model
import argparse
import os
import pandas as pd
from datetime import date
from update_preseason_ratings import PreseasonRatingsUpdater

# update preseason ratings, and models to only use data before the given season (can do the same with weights later) -> skip for now
# predict for all games in given season
# compare to lines

def main(season=2021,
        teamGame_file='data/teamGame.pkl', playerGame_file=None, toiOverlap_file=None,
        preseason_teams_file='data/ratings_preseason_teams.csv',
        preseason_players_file='data/ratings_preseason_players.csv',
        lines_file = 'data/lines.csv'):
    # determine data filenames if not given
    if playerGame_file is None:
        playerGame_file = 'data/playerGame_{}.pkl'.format(str(season))
    if toiOverlap_file is None:
        toiOverlap_file = 'data/toiOverlap_{}.pkl'.format(str(season))

    # read teamGame and playerGame data
    teamGame = pd.read_pickle(teamGame_file)
    playerGame = pd.read_pickle(playerGame_file)

    # only include regular season games
    teamGame = teamGame.loc[teamGame['Playoffs']==0]
    playerGame = playerGame.loc[playerGame['Playoffs']==0]

    # add preseason ratings
    ratings = pd.read_csv(preseason_players_file)
    ratings.columns = ['Player','PlayerID','Position','preseason_xGI60','preseason_xGC60_5v5','preseason_xGP60_5v5',
        'preseason_xGC60_PP','preseason_xGP60_PK','preseason_xGI60_Pens','Season']
    playerGame = playerGame.merge(ratings, on=['Player','PlayerID','Season','Position'], how='left')

    ratings = pd.read_csv(preseason_teams_file)
    colnames = []
    for c in ratings.columns.tolist():
        if c[0]=='x':
            colnames.append('preseason_'+c)
        else:
            colnames.append(c)
    ratings.columns = colnames
    teamGame = teamGame.merge(ratings, on=['Team','Season'], how='left')

    # add inseason ratings
    start_date = date(season, 10, 1)
    end_date = date(season+1, 9, 30)
    #playerGame = inseason_ratings.add_player_inseason_ratings(playerGame, pd.read_pickle(toiOverlap_file), start_date, end_date)
    teamGame = inseason_ratings.add_team_inseason_ratings(teamGame, start_date, end_date)
    print ('preseason ratings added!')

    # calculate features for games model
    df = data_processing.add_game_features(teamGame, playerGame)
    print ('game features added!')
    del teamGame, playerGame
    df = df.loc[df['Season']==season]

    # predict games
    model = games_model.games_model()
    df = model.predict(df)

    # get betting lines data
    lines = pd.read_csv('lines.csv')
    lines = lines.merge(teamGame[['Date','Team','Win']], how='left', on=['Date','Team'])
    lines['ImpliedPct'] = 0.
    lines.loc[lines['Close']<0, 'ImpliedPct'] = lines.loc[lines['Close']<0, 'Close']/(lines.loc[lines['Close']<0, 'Close']-100)
    lines.loc[lines['Close']>0, 'ImpliedPct'] = 100/(100+lines.loc[lines['Close']>0, 'Close'])

    # compare lines to predictions
    #combine game lines with predictions
    df['Opp_winProba'] = 1 - df['winProba']
    home = df[['Date','Team','Opp','winProba']]
    home['Home'] = 1
    away = df[['Date','Opp','Team','Opp_winProba']]
    away['Home'] = 0
    away.columns = ['Date','Team','Opp','winProba','Home']
    tmp_df = pd.concat([home, away], ignore_index=True)
    tmp_df = lines.merge(tmp_df, how='inner', on=['Date','Team'])

    #find all games/teams that would have been bet on
    tmp_df = tmp_df.loc[tmp_df[pct_col]>tmp_df['ImpliedPct'], :]

    #calculate ratio of win to bet, assuming a win
    tmp_df['GainRatio'] = 0.
    tmp_df.loc[tmp_df[line_col]<0, 'GainRatio'] = -100/tmp_df.loc[tmp_df[line_col]<0, line_col]
    tmp_df.loc[tmp_df[line_col]>0, 'GainRatio'] = tmp_df.loc[tmp_df[line_col]>0, line_col]/100.

    #calculate bet amount
    tmp_df['NormBet'] = tmp_df[pct_col]-((1-tmp_df[pct_col])/tmp_df['GainRatio'])

    #calculate net win
    tmp_df['NetWin'] = 0.
    tmp_df.loc[tmp_df['Win']==1, 'NetWin'] = tmp_df.loc[tmp_df['Win']==1, 'NormBet']*tmp_df.loc[tmp_df['Win']==1, 'GainRatio']
    tmp_df.loc[tmp_df['Win']==0, 'NetWin'] = -tmp_df.loc[tmp_df['Win']==0, 'NormBet']

    #aggregate
    print ('Net Win: ' + str(tmp_df['NetWin'].sum()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-season', required=False, help='NHL season to run for. Must pick dates for only one season at a time.')
    args = parser.parse_args()

    main(season=int(args.season))
