import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta5/golf.zip', sep = ',', decimal='.')
# korvataan puuttuvat arvot nollilla
df.fillna(0, inplace=True)
#print(df.iloc[:,30:40])
df['kayttoonottovuosi'] = df['kayttoonottopvm'].astype(str).str[0:4].astype(int)
#print(df['kayttoonottovuosi'])
print('histogrammi käyttöönottovuodesta')
df['kayttoonottovuosi'].plot.hist()
plt.show()

print('histogrammi matkamittarin lukemasta, rajaa pois yli 500 000 kilometrin lukemat')
df2 = df[df['matkamittarilukema'] <= 500000]
df2['matkamittarilukema'].plot.hist()
plt.show()


print('jointplot käyttöönottovuosi<->CO2-päästöt')
#sns.regplot('kayttoonottovuosi', 'Co2', data=df)
#plt.show()
#sns.jointplot('kayttoonottovuosi', 'Co2', data=df, kind='reg')
#plt.show()

print('jointplot omamassa<->suurinNettoteho (rajaa selvästi muista poikkeavat omamassa-arvot pois)')
sns.jointplot('omamassa', 'suurinNettoteho', data=df, kind='reg')
plt.show()

print('violinplot käyttöönottovuodesta käyttövoiman (yksittaisKayttovoima) mukaan. Ota mukaan vain käyttövoiman arvot 1.0 = bensiini, 2.0 = diesel')
df3 = df[df['yksittaisKayttovoima'] <= 2.0]
sns.violinplot(x='yksittaisKayttovoima', y='kayttoonottovuosi', hue='yksittaisKayttovoima', split=True,  data=df3)
plt.show()
