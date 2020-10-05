import re
import pandas as pd
import numpy as np

def calculateLag(df,lag):
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

#Found here: https://github.com/Arturus/kaggle-web-traffic/blob/master/extractor.py

term_pat = re.compile('(.+?):(.+)')
pat = re.compile(
    '(.+)_([a-z][a-z]\.)?((?:wikipedia\.org)|(?:commons\.wikimedia\.org)|(?:www\.mediawiki\.org))_([a-z_-]+?)$')

# Debug output to ensure pattern still works
# print(pat.fullmatch('BLEACH_zh.wikipedia.org_all-accessspider').groups())
# print(pat.fullmatch('Accueil_commons.wikimedia.org_all-access_spider').groups())

def extract_url(source) -> pd.DataFrame:
    """
    Extracts features from url. Features: agent, site, country, term, marker
    :param source: urls
    :return: DataFrame, one column per feature
    """
    if isinstance(source, pd.Series):
        source = source.values
    agents = np.full_like(source, np.NaN)
    sites = np.full_like(source, np.NaN)
    countries = np.full_like(source, np.NaN)
    terms = np.full_like(source, np.NaN)
    markers = np.full_like(source, np.NaN)

    for i in range(len(source)):
        l = source[i]
        match = pat.fullmatch(l)
        assert match, "Non-matched string %s" % l
        term = match.group(1)
        country = match.group(2)
        if country:
            countries[i] = country[:-1]
        site = match.group(3)
        sites[i] = site
        agents[i] = match.group(4)
        if site != 'wikipedia.org':
            term_match = term_pat.match(term)
            if term_match:
                markers[i] = term_match.group(1)
                term = term_match.group(2)
        terms[i] = term

    return pd.DataFrame({
        'agent': agents,
        'site': sites,
        'country': countries,
        'term': terms,
        'marker': markers,
        'Page': source
    })

def shift_visitors(data, shift):
    data['Visitors_shift_' + str(shift)] = 0
    for page in data['Page'].unique():
        df = data.loc[data['Page'] == page]
        data.loc[data['Page'] == page, 'Visitors_shift_' + str(shift)] = df['Visitors'].shift(shift)
    return(data)

def prepareDataXGBoost(df,lag, encoding = 'oneHotEncoding'):
    df = df.fillna(0)
    df_extract = extract_url(df['Page'])
    df = df.set_index('Page')
    df = df.T.rename_axis('Dates')
    df_lag = calculateLag(df,lag).reset_index()
    df_lag = shift_visitors(df_lag,7)
    df_lag = shift_visitors(df_lag,90)
    df_prepared= df_lag.set_index('Page').join(df_extract.set_index('Page'))
    df_prepared = df_prepared.reset_index().set_index(['Dates','Page']).sort_index()
    df_prepared = df_prepared.drop(['term', 'marker'], axis = 1)
    if  encoding == 'oneHotEncoding':
        df_prepared = pd.get_dummies(df_prepared)
    elif encoding == 'label':
        print(df_prepared.columns)
        df_prepared[['agent', 'site', 'country']] = df_prepared[['agent', 'site', 'country']].astype('category')
        df_prepared['agent'] = df_prepared['agent'].cat.codes
        df_prepared['site'] = df_prepared['site'].cat.codes
        df_prepared['country'] = df_prepared['country'].cat.codes
        #df_prepared['term'] = df_prepared['term'].cat.codes
        #df_prepared['marker'] = df_prepared['marker'].cat.codes
    else:
        print("Bad encoding")
    return df_prepared
