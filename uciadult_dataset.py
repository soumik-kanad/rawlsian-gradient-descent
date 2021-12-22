from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import requests
import os

#utility functions
def download_file(url, output_file):
        r = requests.get(url, allow_redirects=True)
        open(output_file, 'wb').write(r.content)

#UCI adults    
class UCIAdultDataset(Dataset):
    """Dataset class for column dataset.
    Args:
    
    """
    url={'train':'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
         'test':'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'}
    
    def __init__(self, split='train', val_ratio=0.05, balance_target_classes=False, binarise_race=False):
        if split not in ['train','test']:
            raise ValueError('Invalid Split %s' % split)
        self.split=split
        self.val_ratio=val_ratio
        self.balance_target_classes=balance_target_classes
        self.binarise_race=binarise_race
        self.download()
        self.read_preprocess()
    
    def download(self):
        self.data_file_path=os.path.join('data','uciadults_'+self.split+'.csv')
        if not os.path.exists(self.data_file_path):
            os.makedirs('data',exist_ok=True)
            download_file(self.url[self.split],self.data_file_path)
            
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return{
            "x": self.data[idx][:-1],
            "y": self.data[idx][-1]
        }
    
    def read_preprocess(self):
        
        if self.split=='train':
            salary_dataset = pd.read_csv(self.data_file_path, header=None, skipinitialspace=True)
        else: #need to skip the first row as it is not relevant
            salary_dataset = pd.read_csv(self.data_file_path, header=None, skipinitialspace=True,skiprows=[0])
            
        df = pd.DataFrame(salary_dataset)
        #add columnnames
        salary_dataset.columns=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
        #remove unknown fields
        df=df.replace('?',np.nan)
        df.dropna(how='any',inplace=True)

        if self.balance_target_classes:
            #balance low income class with high income class
            #https://stackoverflow.com/a/45839920
            g=df.groupby('income')
            df=g.apply(lambda x:x.sample(g.size().min()).reset_index(drop=True))
        
        
        if self.binarise_race:
            #make race!=White as Non-White
            #https://stackoverflow.com/a/31512025
            df.loc[df['race'] != 'White', 'race'] = 'Non-White'

        #make discrete fileds to numbers -> to be converted to onehot
        #except 'education'->convert to number then use as a continuos
        discrete_columns={'education': {'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6, '12th': 7, 'HS-grad': 8, 'Prof-school': 9, 'Assoc-acdm': 10, 'Assoc-voc': 11, 'Some-college': 12, 'Bachelors': 13, 'Masters': 14, 'Doctorate': 15}, 
                          'workclass': {'Private': 0, 'Self-emp-not-inc': 1, 'Self-emp-inc': 2, 'Federal-gov': 3, 'Local-gov': 4, 'State-gov': 5, 'Without-pay': 6, 'Never-worked': 7}, 
                          'marital-status': {'Married-civ-spouse': 0, 'Divorced': 1, 'Never-married': 2, 'Separated': 3, 'Widowed': 4, 'Married-spouse-absent': 5, 'Married-AF-spouse': 6}, 
                          'occupation': {'Tech-support': 0, 'Craft-repair': 1, 'Other-service': 2, 'Sales': 3, 'Exec-managerial': 4, 'Prof-specialty': 5, 'Handlers-cleaners': 6, 'Machine-op-inspct': 7, 'Adm-clerical': 8, 'Farming-fishing': 9, 'Transport-moving': 10, 'Priv-house-serv': 11, 'Protective-serv': 12, 'Armed-Forces': 13}, 
                          'relationship': {'Wife': 0, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 3, 'Other-relative': 4, 'Unmarried': 5}, 
                          'race': {'White': 0, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 2, 'Other': 3, 'Black': 4, 'Non-White':1}, 
                          'sex': {'Female': 0, 'Male': 1}, 
                          #'native-country': {'United-States': 0, 'Cambodia': 1, 'England': 2, 'Puerto-Rico': 3, 'Canada': 4, 'Germany': 5, 'Outlying-US(Guam-USVI-etc)': 6, 'India': 7, 'Japan': 8, 'Greece': 9, 'South': 10, 'China': 11, 'Cuba': 12, 'Iran': 13, 'Honduras': 14, 'Philippines': 15, 'Italy': 16, 'Poland': 17, 'Jamaica': 18, 'Vietnam': 19, 'Mexico': 20, 'Portugal': 21, 'Ireland': 22, 'France': 23, 'Dominican-Republic': 24, 'Laos': 25, 'Ecuador': 26, 'Taiwan': 27, 'Haiti': 28, 'Columbia': 29, 'Hungary': 30, 'Guatemala': 31, 'Nicaragua': 32, 'Scotland': 33, 'Thailand': 34, 'Yugoslavia': 35, 'El-Salvador': 36, 'Trinadad&Tobago': 37, 'Peru': 38, 'Hong': 39, 'Holand-Netherlands': 40}, 
                          'native-country': {'United-States': 1, 'Cambodia': 0, 'England': 0, 'Puerto-Rico': 0, 'Canada': 0, 'Germany': 0, 'Outlying-US(Guam-USVI-etc)': 0, 'India': 0, 'Japan': 0, 'Greece': 0, 'South': 0, 'China': 0, 'Cuba': 0, 'Iran': 0, 'Honduras': 0, 'Philippines': 0, 'Italy': 0, 'Poland': 0, 'Jamaica': 0, 'Vietnam': 0, 'Mexico': 0, 'Portugal': 0, 'Ireland': 0, 'France': 0, 'Dominican-Republic': 0, 'Laos': 0, 'Ecuador': 0, 'Taiwan': 0, 'Haiti': 0, 'Columbia': 0, 'Hungary': 0, 'Guatemala': 0, 'Nicaragua': 0, 'Scotland': 0, 'Thailand': 0, 'Yugoslavia': 0, 'El-Salvador': 0, 'Trinadad&Tobago': 0, 'Peru': 0, 'Hong': 0, 'Holand-Netherlands': 40}, 
                          'income': {'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1}} #train has no '.' in income
        for col in discrete_columns:
            df[col] = df[col].map(discrete_columns[col]).astype(int)
        print(df.head())
        print(df['income'].sum(), len(df['income']))
        #normalise continuous fileds
        continuous_columns={'age': (38.437901995888865, 13.134664776855985), 'fnlwgt': (189793.83393011073, 105652.971528519), 'education-num': (10.12131158411246, 2.549994918856736), 'capital-gain': (1092.0078575691268, 7406.346496683503), 'capital-loss': (88.37248856176646, 404.29837048637575), 'hours-per-week': (40.93123798156621, 11.979984229274882), 'education': (9.92676215105099, 2.9812095549490087)}
        for col in continuous_columns:
            mu=continuous_columns[col][0]
            std=continuous_columns[col][1]
            df[col]=(df[col]-mu)/std
        self.discrete_columns=discrete_columns
        self.continuous_columns=continuous_columns
        #covert data tto np array
        data=[]
        self.col_pos={}
        # for col in df.columns:
        #     if col in continuous_columns:
        #         data.append( np.expand_dims(df[col].to_numpy(),-1) )
        #     elif col == 'income':
        #         data.append( np.expand_dims(df[col].to_numpy(),-1))
        #     else:
        #         data.append(self.one_hot(col,df[col].to_numpy()))
        # self.data=np.concatenate(data, axis=-1)       

        for col in df.columns:
            #convert to one hot if required
            if col in continuous_columns:
                _new_data= np.expand_dims(df[col].to_numpy(),-1) 
            elif col == 'income':
                _new_data = np.expand_dims(df[col].to_numpy(),-1)
            else:
                _new_data = self.one_hot(col,df[col].to_numpy())

            #concatenate
            if len(data)==0:
                start=0
                data=_new_data
            else:
                start=data.shape[-1]-1
                data=np.concatenate([data,_new_data], axis=-1)

            #save col pos
            end=data.shape[-1]-1
            self.col_pos[col]= (start,end)
        
        self.data=data

        #assign as class varibales
        self.df=df
        self.num_features=self.data.shape[-1]-1 #-1 for removing the label
       
    def one_hot(self,col,value):
        n_values = len(self.discrete_columns[col].keys())
        return np.eye(n_values)[value]