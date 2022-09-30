# TODONOW: team roster configs

# ORDER TO RUN:
#1. update_data.py for each season (includes data scraping)
#2. use python in command line to do update_preseason_ratings.py. only necessary for a new season
#3. predict_today.py

# TODO: only run elo updates for new games
# TODO: predictions only work for one day at a time
# TODO: scraper may have included some preseason games in the data

import data_scraper
import data_processing
import inseason_ratings
import games_model
import argparse
import os
import pandas as pd
import numpy as np
import datetime
from datetime import date
from update_preseason_ratings import PreseasonRatingsUpdater

def main(tofile, start_date=None, end_date=None, season=2022,
        teamGame_file='data/teamGame.pkl', playerGame_file=None, toiOverlap_file=None,
        preseason_teams_file='data/ratings_preseason_teams.csv',
        preseason_players_file='data/ratings_preseason_players.csv',
        out_file=None, savetrain=False, train_file=None):
    # use yesterday if no dates are given
    if start_date is None:
        start_date = date.today().strftime('%Y-%m-%d')
    if end_date is None:
        end_date = date.today().strftime('%Y-%m-%d')

    # determine data filenames if not given
    if playerGame_file is None:
        playerGame_file = 'data/playerGame_{}.pkl'.format(str(season))
    if toiOverlap_file is None:
        toiOverlap_file = 'data/toiOverlap_{}.pkl'.format(str(season))
    if tofile and out_file is None:
        out_file = 'data/gamePredictions_{}.csv'.format(str(season))
    if savetrain and train_file is None:
        out_file = 'data/gameTrain_{}.csv'.format(str(season))

    # get schedule of games to predict
    schedule = data_scraper.scrape_schedule(start_date, end_date)
    print ('done scraping!')

    # read teamGame and playerGame data
    teamGame = pd.read_pickle(teamGame_file)
    try:
        playerGame = pd.read_pickle(playerGame_file)
        seasonFirstDay = False
    except FileNotFoundError:
        # there have not been any games yet this season, so playerGame doesn't exist yet
        playerGame = pd.DataFrame(columns=['Player','PlayerID','TOI_5v5'])
        seasonFirstDay = True
        pass

    # combine with scheduled games
    teamGame = data_processing.add_scheduled_games(teamGame, schedule, season)
    ### TODONOW: add projected lineup to playerGame, need ['Player','PlayerID','Position','Season','Date','Playoffs']
    rosters = pd.DataFrame()
    ids = pd.read_pickle('data/playerGame_{}.pkl'.format(str(int(season)-1)))[['Player','PlayerID']]
    ids = pd.concat([playerGame[['Player','PlayerID']], ids], ignore_index=True)
    ids = ids.groupby('Player', as_index=False).max()
    allTeams = pd.concat([schedule[['home_team']].rename(columns={'home_team':'Team'}),\
        schedule[['away_team']].rename(columns={'away_team':'Team'})])
    allTeams['Team'] = allTeams['Team'].replace({'PHX':'ARI', 'S.J':'SJS', 'L.A':'LAK', 'T.B':'TBL', 'N.J':'NJD'})
    for t in allTeams['Team'].unique().tolist():
        try:
            roster = pd.read_json('configs/rosters/{}.json'.format(t))
            roster = roster.merge(ids, on='Player', how='left')
            if len(roster.loc[roster['PlayerID'].isnull()].index)>0:
                print ('Missing PlayerIDs:')
                print (roster.loc[roster['PlayerID'].isnull()])
            roster['Season'] = int(season)
            roster['Date'] = start_date
            roster['DateInt'] = roster['Date'].str.replace('-','').astype(np.int64)
            roster['Playoffs'] = 0
            roster['Game_Id'] = schedule.loc[((schedule['home_team']==t)|(schedule['away_team']==t))&(schedule['date']==start_date),'game_id'].max()
            roster['Team'] = t
            playerGame = pd.concat([playerGame, roster], ignore_index=True)
            teamGame.loc[(teamGame['Team']==t)&(teamGame['Date']==start_date), 'StartingGoalie_Id'] = roster.loc[roster['Position']=='G', 'PlayerID'].max()
        except FileNotFoundError:
            print ('No roster found for {}, ignoring this team'.format(t))
            teamGame = teamGame.loc[~((teamGame['Team']==t)&(teamGame['Date'].isin(schedule['date'].unique())))]
            pass


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
    if seasonFirstDay:
        playerGame = inseason_ratings.add_player_inseason_ratings(playerGame, pd.DataFrame(columns=['Player_x','Player_Id_x','Strength','Game_Id']), start_date, end_date)
    else:
        playerGame = inseason_ratings.add_player_inseason_ratings(playerGame, pd.read_pickle(toiOverlap_file), start_date, end_date)
    teamGame = inseason_ratings.add_team_inseason_ratings(teamGame, start_date, end_date)
    print ('preseason ratings added!')

    # calculate features for games model
    df = data_processing.add_game_features(teamGame, playerGame)
    print ('game features added!')
    del teamGame, playerGame
    df = df.loc[df['Date']>=start_date]
    df = df.loc[df['Date']<=end_date]

    # save training data to file
    if savetrain:
        df.to_csv(train_file, index=False)

    # predict today's games
    model = games_model.games_model()
    df = model.predict(df)

    if tofile:
        df.to_csv(out_file, index=False)
    else:
        print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dates', nargs=2, required=False, help='First date is start date, second date is end date. Both dates must be given if the -dates flag is used')
    parser.add_argument('-season', required=False, help='NHL season to run for. Must pick dates for only one season at a time.')
    parser.add_argument('--tofile', help='Saves output to csv if flag is added, otherwise will just print to stdout', action='store_true')
    parser.add_argument('--savetrain', help='Saves training data to csv if flag is added', action='store_true')
    args = parser.parse_args()

    if args.dates is None:
        main(tofile=args.tofile, savetrain=args.savetrain)
    else:
        main(tofile=args.tofile, start_date = args.dates[0], end_date = args.dates[1], season=args.season, savetrain=args.savetrain)
