import pandas as pd

dfPyorat = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/pyorat.txt', sep = ',', decimal='.')
dfSaa = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/helsinki2017.csv', sep = ',', decimal='.')

dfPyorat.rename(columns={'Month': 'Kk', 'Day': 'Pv', 'Hour':'Klo'}, inplace=True)
dfSaa['Klo'] = (dfSaa['Klo'].str.split(':').str.get(0)).astype(int)

yhdistettävät = ['Kk', 'Pv', 'Klo']
dfYhd2 = pd.merge(dfPyorat, dfSaa, how = 'outer', on = yhdistettävät) #dfPyorat.combine_first(dfSaa)
#print(dfYhd2[yhdistettävät].sort_values(['Kk', 'Pv', 'Klo'], ascending=False).head(50))
print(dfYhd2)