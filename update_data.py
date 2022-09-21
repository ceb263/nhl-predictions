import data_scraper
import data_processing
import argparse
import os
import pandas as pd
import datetime
from datetime import date, timedelta

def main(start_date=None, end_date=None, season=2022, pbp_data_file=None, shifts_data_file=None, preseason_ratings_file='data/ratings_preseason.csv',
    output_playerGame=None, output_toiOverlap=None, output_teamGame='data/teamGame.pkl', replace=False, teamGameReplace=False):
    # use yesterday if dates are not given
    if start_date is None:
        start_date = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    # determine data filenames if not given
    if pbp_data_file is None:
        pbp_data_file = 'data/pbp_{}.pkl'.format(str(season))
    if shifts_data_file is None:
        shifts_data_file = 'data/shifts_{}.pkl'.format(str(season))
    if output_playerGame is None:
        output_playerGame = 'data/playerGame_{}.pkl'.format(str(season))
    if output_toiOverlap is None:
        output_toiOverlap = 'data/toiOverlap_{}.pkl'.format(str(season))

    # scrape data
    data_scraper.scrape_dates(start_date, end_date, pbp_data_file=pbp_data_file, shifts_data_file=shifts_data_file)

    # get pbp data
    pbp = pd.read_pickle(pbp_data_file)
    shifts = pd.read_pickle(shifts_data_file)

    # filter for only dates between start_date and end_date (inclusive)
    shifts = shifts.loc[shifts['Date']>=start_date]
    pbp = pbp.loc[pbp['Date']>=start_date]
    shifts = shifts.loc[shifts['Date']<=end_date]
    pbp = pbp.loc[pbp['Date']<=end_date]

    # process data and output
    pbp, shots = data_processing.get_shots_data(pbp, season)
    pbp = data_processing.add_xG_to_pbp(pbp, shots)
    del shots
    playerGame, toi_overlap = data_processing.aggregate_player_data(pbp, shifts)
    del shifts
    if os.path.isfile(output_playerGame) and not replace:
        playerGame = pd.concat([pd.read_pickle(output_playerGame), playerGame], ignore_index=True)
    playerGame.to_pickle(output_playerGame)
    del playerGame
    if os.path.isfile(output_toiOverlap) and not replace:
        toi_overlap = pd.concat([pd.read_pickle(output_toiOverlap), toi_overlap], ignore_index=True)
    toi_overlap.to_pickle(output_toiOverlap)
    del toi_overlap

    if os.path.isfile(output_teamGame) and not teamGameReplace:
        prev_teamGame = pd.read_pickle(output_teamGame)
    else:
        prev_teamGame = None
    teamGame = data_processing.aggregate_team_data(pbp, prev_teamGame)
    teamGame.to_pickle(output_teamGame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dates', nargs=2, required=False, help='First date is start date, second date is end date. Both dates must be given if the -dates flag is used')
    parser.add_argument('-season', required=False, help='NHL season to run for. Must pick dates for only one season at a time.')
    parser.add_argument('--replace', help='Replaces any existing data if flag is included', action='store_true')
    parser.add_argument('--teamGameReplace', help='Replaces any existing data for teamGame only if flag is included', action='store_true')
    args = parser.parse_args()

    if args.dates is None:
        main(replace=args.replace)
    else:
        main(start_date = args.dates[0], end_date = args.dates[1], season = args.season, replace=args.replace, teamGameReplace=args.teamGameReplace)
