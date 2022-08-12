import pickle
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score

scriptdir = os.path.dirname(__file__)
config_file = os.path.join(scriptdir, 'models/games_config.json')
with open(config_file, 'r') as f:
    games_config = json.load(f)

class games_model(object):
    def __init__(self):
        self.scaler_file = os.path.join(scriptdir, 'models/', games_config['scaler_file'])
        self.model_file = os.path.join(scriptdir, 'models/', games_config['model_file'])

    def predict(self, df):
        # makes game predictions for games in the provided dataframe

        # filter for only home regular season games
        if ['Home'] in df.columns:
            df = df.loc[df['Home']==1]
        if ['Playoffs'] in df.columns:
            df = df.loc[df['Playoffs']==0]

        # create input feature array
        X = df[games_config['features']].values
        scaler = pickle.load(open(self.scaler_file, 'rb'))
        X = scaler.transform(X)

        # make predictions
        model = pickle.load(open(self.model_file, 'rb'))
        preds = model.predict_proba(X)[:,1]

        df = df[['Date','Team','Opp']]
        df['winProba'] = preds

        return df

    def train(self, df):
        # retrains goals model based on given data

        #df = df.loc[~df[features].isnull().max(1)]

        X = df[games_config['features']].values
        y = df['Win'].values
        del df

        # apply scaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        pickle.dump(scaler, open(self.scaler_file, 'wb'))

        model = LogisticRegression(max_iter=10000)
        model.fit(X, y)
        pickle.dump(model, open(self.model_file, 'wb'))

        preds = model.predict_proba(X)[:,1]
        print ('LogLoss score on train set: {}'.format(str(log_loss(y, preds))))
        print ('AUC score on train set: {}'.format(str(roc_auc_score(y, preds))))
