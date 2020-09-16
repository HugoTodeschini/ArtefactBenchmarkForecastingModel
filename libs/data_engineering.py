import pandas as pd

def calculerLag(df,lag):
    df_lag = pd.DataFrame()
    for column in df.columns:
        df_column = pd.DataFrame(df[column])
        for i in range(lag):
            df_column['diff' + str(i+1)] = 0
            df_column['diff' + str(i+1)] = df_column[column].diff(i+1)
        
        df_column['Page'] = column
        df_column['Visitors'] = df_column[column]
        df_column = df_column.drop([column],axis= 1)
        df_lag.append(df_column)
        df_lag = pd.concat([df_lag, df_column], ignore_index=False)
    return(df_lag)

