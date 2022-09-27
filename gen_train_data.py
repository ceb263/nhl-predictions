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
    playerGame = inseason_ratings.add_player_inseason_ratings(playerGame, pd.read_pickle(toiOverlap_file), start_date, end_date)
    teamGame = inseason_ratings.add_team_inseason_ratings(teamGame, start_date, end_date)
    print ('preseason ratings added!')

    # calculate features for games model
    df = data_processing.add_game_features(teamGame, playerGame)
    print ('game features added!')
    del playerGame
    df = df.loc[df['Season']==season]
    print (df.loc[df['Goalie_xGI60'].isnull()].head()[['Date','Team','StartingGoalie']])

    df.to_csv('data/gameTrain_{}.csv'.format(str(season)), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-season', required=False, help='NHL season to run for. Must pick dates for only one season at a time.')
    args = parser.parse_args()

    main(season=int(args.season))
