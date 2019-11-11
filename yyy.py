def column(df):
    for column in df:
        print(column,':', df[column].nunique())


