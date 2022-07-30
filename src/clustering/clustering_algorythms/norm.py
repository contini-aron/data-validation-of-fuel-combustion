from sklearn import preprocessing
import pandas as pd

def normalize(df: pd.DataFrame, minmax:bool = False)->pd.DataFrame:
    """
    normalizes data if minmax is true uses minmax scaler
    """
    names = df.columns
    if not minmax:
        retval= pd.DataFrame(preprocessing.normalize(df, axis=0), columns=names)
        print(retval)
        return retval
    else:
        scaler = preprocessing.MinMaxScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=names)
