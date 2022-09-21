# TODO: weight recent seasons higher?

import pickle
import os
import json
import lightgbm as lgb
import data_processing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score

scriptdir = os.path.dirname(__file__)
config_file = os.path.join(scriptdir, 'configs/xG_config.json')
with open(config_file, 'r') as f:
    xG_config = json.load(f)

class xG_model(object):
    def __init__(self):
        self.mean_encodings_file = os.path.join(scriptdir, 'models/', xG_config['mean_encodings_file'])
        self.scaler_file = os.path.join(scriptdir, 'models/', xG_config['scaler_file'])
        self.model_file = os.path.join(scriptdir, 'models/', xG_config['model_file'])

    def predict(self, df):
        # makes xG predictions for shots in the provided dataframe

        # create mean encoded columns for categorical variables
        mean_encodings = pickle.load(open(self.mean_encodings_file, 'rb'))
        for i, colname in xG_config['mean_encodings'].items():
            df[colname+'_meanEnc'] = df[colname].map(mean_encodings[i])

        # create input feature array
        X = df[xG_config['features']].values
        scaler = pickle.load(open(self.scaler_file, 'rb'))
        X = scaler.transform(X)

        # make predictions
        model = pickle.load(open(self.model_file, 'rb'))
        preds = model.predict_proba(X)[:,1]

        return preds

    def train(self, max_season=2020):
        # retrains xG model

        # read data and process into just shots for xG model
        df = data_processing.get_shots_data(pd.read_pickle('data/pbp_2012.pkl'), 2012)[1]
        for season in range(2013, max_season+1):
            df = pd.concat([df, data_processing.get_shots_data(pd.read_pickle('data/pbp_{}.pkl'.format(str(season))), season)[1]], ignore_index=True)

        # get mean encodings
        mean_codes_strength = df.groupby(['Strength'])['goal'].mean().to_dict()
        mean_codes_zone = df.groupby(['Ev_Zone'])['goal'].mean().to_dict()
        mean_codes_type = df.groupby(['Type'])['goal'].mean().to_dict()
        mean_codes_prevEvent = df.groupby(['prev_Event'])['goal'].mean().to_dict()
        mean_codes_shotCategory = df.groupby(['ShotCategory'])['goal'].mean().to_dict()
        mean_codes = {
            'shotCategory' : mean_codes_shotCategory,
            'strength' : mean_codes_strength,
            'zone' : mean_codes_zone,
            'type' : mean_codes_type,
            'prevEvent' : mean_codes_prevEvent
        }
        pickle.dump(mean_codes, open(self.mean_encodings_file, 'wb'))

        df['Strength_meanEnc'] = df['Strength'].map(mean_codes_strength)
        df['Ev_Zone_meanEnc'] = df['Ev_Zone'].map(mean_codes_zone)
        df['Type_meanEnc'] = df['Type'].map(mean_codes_type)
        df['prev_Event_meanEnc'] = df['prev_Event'].map(mean_codes_prevEvent)
        df['ShotCategory_meanEnc'] = df['ShotCategory'].map(mean_codes_shotCategory)

        X = df[xG_config['features']].values
        y = df['goal'].values
        del df

        # apply scaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        pickle.dump(scaler, open(self.scaler_file, 'wb'))

        model = lgb.LGBMClassifier(max_depth=10, min_child_samples=200, random_state=26, n_estimators=130)
        model.fit(X, y)
        pickle.dump(model, open(self.model_file, 'wb'))

        preds = model.predict_proba(X)[:,1]
        print ('LogLoss score on train set: {}'.format(str(log_loss(y, preds))))
        print ('AUC score on train set: {}'.format(str(roc_auc_score(y, preds))))
