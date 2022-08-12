import hockey_scraper
import pandas as pd
import os
import datetime
from datetime import date, timedelta

def scrape_dates(start_date=None, end_date=None, pbp_data_file='data/pbp_2022.pkl', shifts_data_file='data/shifts_2022.pkl', season_start='2022-10-07', replace=False):
    # start_date and end_date specify the date range to scrape data for
    # if start_date and end_date are None, data will be scraped that is after any existing data (or the beginning of the season if there is none), up to yesterday
    # pbp_data_file and shifts_data_file specify where data should be saved (assumed to be .pkl files)
    # season_start is a string (YYYY-MM-DD format) with the first date of games for the current NHL season
    # replace will replace any data in the data files with the newly scraped data, if the files exist. if replace=False, new data will be appended
    # if the files do not exist yet, or are empty, then the replace parameter is meaningless

    # read existing pbp data for this season, if it exists
    if os.path.isfile(pbp_data_file):
        df_pbp = pd.read_pickle(pbp_data_file)
    else:
        replace = True

    # set start date to be one day after any existing data, or the beginning of the season if there is no existing data
    if start_date is None:
        if df_pbp.empty or replace:
            start_date = season_start
            replace = True
        else:
            start_date = (datetime.datetime.srtptime(df_pbp['Date'].max(), '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

    # if end_date is not given, set it to yesterday
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    # scrape data
    scrape_dict = hockey_scraper.scrape_date_range(start_date, end_date, True, data_format='Pandas')

    # get dataframes
    df_pbp_new = scrape_dict['pbp']
    df_shifts_new = scrape_dict['shifts']
    del scrape_dict #save some memory

    # append to existing data, if it exists. then output data to files
    if replace:
        df_pbp_new.to_pickle(pbp_data_file)
        df_shifts_new.to_pickle(shifts_data_file)
    else:
        df_pbp = pd.concat([df_pbp, df_pbp_new], ignore_index=True)
        del df_pbp_new
        df_pbp.to_pickle(pbp_data_file)
        del df_pbp

        df_shifts = pd.read_pickle(shifts_data_file)
        df_shifts = pd.concat([df_shifts, df_shifts_new], ignore_index=True)
        df_shifts.to_pickle(shifts_data_file)

def scrape_schedule(start_date=date.today().strftime('%Y-%m-%d'), end_date=date.today().strftime('%Y-%m-%d')):
    # get today's games by default, can get schedule for other dates between start_date and end_date, inclusive
    return hockey_scraper.scrape_schedule(start_date, end_date)
