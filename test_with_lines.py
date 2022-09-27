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

def main(season=2021, lines_file='data/lines.csv', scaler_file=None, model_file=None):
    # predict games
    if model_file is None or scaler_file is None:
        model = games_model.games_model(scaler_file='models/test_scaler.pkl', model_file='models/test_lr.pkl')
        model.train(max_season=season-1)
        df = model.predict(pd.read_csv('data/gameTrain_{}.csv'.format(str(season))))
        os.remove('models/test_scaler.pkl')
        os.remove('models/test_lr.pkl')
    else:
        model = games_model.games_model(scaler_file=scaler_file, model_file=model_file)
        df = model.predict(pd.read_csv('data/gameTrain_{}.csv'.format(str(season))))

    # get betting lines data
    lines = pd.read_csv(lines_file)
    lines = lines.merge(df[['Date','Team','Win']], how='left', on=['Date','Team'])
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
    tmp_df = tmp_df.loc[tmp_df['winProba']>tmp_df['ImpliedPct'], :]

    #calculate ratio of win to bet, assuming a win
    tmp_df['GainRatio'] = 0.
    tmp_df.loc[tmp_df['Close']<0, 'GainRatio'] = -100/tmp_df.loc[tmp_df['Close']<0, 'Close']
    tmp_df.loc[tmp_df['Close']>0, 'GainRatio'] = tmp_df.loc[tmp_df['Close']>0, 'Close']/100.

    #calculate bet amount
    tmp_df['NormBet'] = tmp_df['winProba']-((1-tmp_df['winProba'])/tmp_df['GainRatio'])

    #calculate net win
    tmp_df['NetWin'] = 0.
    tmp_df.loc[tmp_df['Win']==1, 'NetWin'] = tmp_df.loc[tmp_df['Win']==1, 'NormBet']*tmp_df.loc[tmp_df['Win']==1, 'GainRatio']
    tmp_df.loc[tmp_df['Win']==0, 'NetWin'] = -tmp_df.loc[tmp_df['Win']==0, 'NormBet']

    #aggregate
    print ('Net Win: ' + str(tmp_df['NetWin'].sum()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-season', required=False, help='NHL season to run for. Must pick dates for only one season at a time.')
    parser.add_argument('-scaler_file', required=False, help='File location of scaler to use for data')
    parser.add_argument('-model_file', required=False, help='File location of model to use for prediction')
    args = parser.parse_args()

    main(season=int(args.season), scaler_file=args.scaler_file, model_file=args.model_file)
