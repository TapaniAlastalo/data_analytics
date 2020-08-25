import pandas as pd

df = pd.read_csv('http://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta3/saajkl20200118.csv', sep = ',')

df.loc[(df['Lumensyvyys'] < 0), 'Lumensyvyys'] = 0
df.loc[(df['Lumensyvyys'].isnull()), 'Lumensyvyys'] = df.shift(1)['Lumensyvyys']
#print(df.sort_values(['Kk'], ascending=False).head(15))

df['Talvi'] = 0
df.loc[(df['Kk'] >= 11 ), 'Talvi'] = df['Vuosi']
df.loc[(df['Kk'] == 1 ) & (df['Pv'] <= 18 ), 'Talvi'] = df['Vuosi'] -1

print("Talvet")
talvet = df[(df['Talvi'] > 0)]
talvet['Lumipäivä'] = None
talvet.loc[(df['Lumensyvyys'] > 0 ), 'Lumipäivä'] = True
#print(talvet)

alkutalvet = pd.DataFrame(talvet['Lumensyvyys'].groupby(talvet['Talvi']).sum())

alkutalvet = alkutalvet.sort_values(['Lumensyvyys'], ascending=False)
alkutalvet.insert(0, 'Sija', range(1, 1 + len(alkutalvet)))

alkutalvet['Lumipäiviä'] = talvet['Lumipäivä'].groupby(talvet['Talvi']).count()
alkutalvet['Maksimi'] = talvet['Lumensyvyys'].groupby(talvet['Talvi']).max()

alkutalvet = alkutalvet.sort_values(['Talvi'], ascending=True)
print(alkutalvet)