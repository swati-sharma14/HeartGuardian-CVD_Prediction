from __future__ import print_function, absolute_import

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



class MortalityPredictor:
    def __init__(self,mode,path,n_split=5,standardize=True):
        
        '''
        INPUTS:
            > mode: Base model used (0, 1, 2)
                    0 -> AdaBoost
                    1 -> Logistic Regression
                    2 -> Random Forests
            > n_split:  Number of splits for stratified k-fold splitting
                        Default = 5
            > path: Absolute path to the dataset to be used
            > standardize (bool): Specify whether to standardize dataset or not
                                  Default = True
        '''

        self.path                               = path
        self.standardize                        = standardize

        self.data, self.time, self.event        = self.load_dataset()
        self.models                             = self.init_models()
        self.mode                               = mode
        
        self.n                                  = n_split
        self.skf                                = StratifiedKFold(n_splits=self.n)

        self.preds                              = []
        self.ccdfs                              = []
        self.scores                             = []


    def init_models(self):
        ada      = AdaBoostClassifier(n_estimators=100,learning_rate=1.0,random_state=42)
        lr       = LogisticRegression(dual=False,tol=1e-5,C=0.8,fit_intercept=True,solver='newton-cholesky',penalty='l2')
        rf       = RandomForestClassifier(n_estimators=100,criterion='log_loss',max_depth=30,min_samples_split=5,min_samples_leaf=5,max_features=0.5)
        
        return [ada,lr,rf]


    def load_dataset(self):
        raw = pd.read_csv(self.path)

        ## Separate the required variables and use one-hot-encoding for the categorical variables
        raw_cat         = raw[['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS',
                                'educ', 'PREVCHD', 'PREVAP', 'PREVMI',
                                'PREVSTRK', 'PREVHYP']]
        raw_num         = raw[['TOTCHOL', 'AGE', 'SYSBP', 'DIABP',
                                'CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE']]

        raw_cat_1       = pd.get_dummies(raw_cat)
        raw_num_1       = raw_num
        cov             = pd.concat([raw_cat_1,raw_num_1],axis=1)

        ## Replace all null values in raw data
        cov             = self.replace_nulls(cov)
        cols            = cov.columns

        ## Standardize raw
        if(self.standardize):
            cov         = StandardScaler().fit_transform(cov)
        
        ## Time of Event and Event Labels
        cov_1           = pd.DataFrame(cov,columns=cols)
        time            = pd.DataFrame((raw['TIMEDTH'] - raw['TIME']).values,columns=["TIME"])
        event           = (raw['DEATH'])

        return cov_1, time, event


    def replace_nulls(self,dataset):
        ## educ null replaced by 0
        dataset.educ.fillna(0,inplace=True)

        ## CIGPDAY null replaced by median (only for CURSMOKE==1)
        dataset.CIGPDAY.fillna(dataset.CIGPDAY.where(dataset.CURSMOKE==1).median(),inplace=True)

        ## TOTCHOL, BMI null replaced by median
        dataset.TOTCHOL.fillna(dataset.TOTCHOL.median(),inplace=True)
        dataset.BMI.fillna(dataset.BMI.median(),inplace=True)

        ## BPMEDS null replaced by 0 (no medication)
        dataset.BPMEDS.fillna(0,inplace=True)

        ## GLUCOSE null replaced by median
        dataset.GLUCOSE.fillna(dataset.GLUCOSE.median(),inplace=True)

        ## HEARTRTE null replaced by median
        dataset.HEARTRTE.fillna(dataset.HEARTRTE.median(),inplace=True)

        return dataset

    
    def ctd_score(self):
        self.d       = 0
        self.n       = 0
        self.ai      = 0

        self.surv    = self.surv[np.reshape(self.labs, (-1)) == 0, :]
        self.times   = self.times[np.reshape(self.labs, (-1)) == 0, :]
        self.labs    = self.labs[np.reshape(self.labs, (-1)) == 0]

        for i in range(self.times.shape[0]):
            self.ai       = (self.times > self.times[i]) * 1
            self.n       += np.sum(self.ai * (self.surv[i, np.where(self.time_list==self.times[i])[0]] < self.surv[:, np.where(self.time_list==self.times[i])[0]]))
            self.d       += np.sum(self.ai)
            
        self.score   = (self.n/self.d)

    
    def train_models(self):
        '''
        Approach for Mortality Prediction followed here is described by the pseudocode snippet
        attached in the README file for the repository. The approach takes each time instance
        (say t) for the current training data, and trains a new model corresponding to that
        instance with labels defined as 0 or 1 where 0 indicates the sample did not exhibit
        the event of interest uptil time instance t, and 1 indicates the sample exhibited the
        event of interest on or before time t.
        '''

        ## Train-test split
        data_train, data_test       = self.data.iloc[self.train], self.data.iloc[self.test]
        event_train, event_test     = self.event.iloc[self.train], self.event.iloc[self.test]
        time_train, time_test       = self.time.iloc[self.train], self.time.iloc[self.test]

        ## Censored samples are to be ignored
        flag_tr              = event_train==0
        flag_ts              = event_test==0

        self.data_train      = np.array(data_train[flag_tr])
        self.data_test       = np.array(data_test[flag_ts])

        self.time_train      = np.array(time_train[flag_tr]).reshape(time_train[flag_tr].shape[0])
        self.time_test       = np.array(time_test[flag_ts]).reshape((time_test[flag_ts].shape[0],1))

        self.event_train     = np.array(event_train[flag_tr]).reshape(event_train[flag_tr].shape[0])
        self.event_test      = np.array(event_test[flag_ts]).reshape(event_test[flag_ts].shape[0])
        
        ## All unique values of the time of event from the training set
        self.time_list       = np.sort(np.unique(self.time_train))
        self.len             = float(self.time_list.shape[0])
        self.preds           = []

        ## Train a model over each unique time instant
        for t in self.time_list:
            ## Model corresponding to chosen mode
            self.model  = self.models[self.mode]

            ## Empty df of labels corresponding to each sample
            Y           = np.zeros(self.time_train.shape[0])
            
            label       = self.time_train <= t
            Y[label]    = 1
            
            X           = self.data_train
            Y           = Y.reshape(Y.shape[0])

            ## Train the model
            self.model.fit(X=X,y=Y)
            self.preds.append(np.array(self.model.predict(self.data_test)))

        self.preds     = np.array(self.preds).T


    def run(self):
        ## k-fold splitting
        for train_ind, test_ind in self.skf.split(self.data,self.event):
            self.train      = train_ind
            self.test       = test_ind

            ## Train Tmax+1 models for each time instance in the current set
            self.train_models()

            ## Convert all predictions for current split to ccdf
            self.convert_to_ccdf()

            ## Calculate ctd-score for current split
            self.get_score()

    
    def convert_to_ccdf(self):
        self.ccdf      = np.array([[1.000000]*int(self.len)]*self.data_test.shape[0])
        
        for i,pred in enumerate(self.preds):
            cnts          = np.cumsum(pred) + 0.00001*i
            self.ccdf[i] -= cnts/self.len

        self.ccdfs.append(self.ccdf)

        
    def get_score(self):
        self.surv       = self.ccdf
        self.times      = self.time_test
        self.labs       = self.event_test

        self.ctd_score()
        self.scores.append(self.score)