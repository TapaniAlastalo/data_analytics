import pandas as pd

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/kysely.csv', sep = ',', decimal='.')
# korvataan puuttuvat arvot nollilla
df.fillna(0, inplace=True)

print('\nlukumäärät siten, että riveillä perhesuhteet, sarakkeissa tyytyväisyys johtoon (lukumäärät, eli kuinka monta perheellistä on vastannut "tyytyväisyys johtoon"-kohtaan ykkösen jne')
print(pd.crosstab(df['perhetilanne'],  df['TyytJohto']))

print('\nprosenttiosuudet siten, että riveillä koulutus tekstinä (esim peruskoulu), sarakkeissa tyytyväisyys työtehtäviin ( prosenttiosuudet 1 desimaalilla riveittäin eli esim montako % vain peruskoulun käyneistä vastaa 1 jne.)')
print(pd.crosstab(df['koulutus'],  df['TyytTyöteht'], normalize = 'index').applymap("{:.1%}".format).stack())

print('\nkeskiarvot tyytyväisyyksistä johtoon ja työtehtäviin siten, että riveillä on ikäluokka 20-29, 30-39, ... 60-69 ja (eri sarakkeissa) tyytyväisyydet johtoon ja työtehtäviin')
df2 = df
df2['Ikäluokka'] = df['ikä'].astype(str).str[0] + '0-' + df['ikä'].astype(str).str[0] + '9'
print(df2.pivot_table(['TyytJohto', 'TyytTyöteht'], index=['Ikäluokka']).applymap('{:,.1f}'.format)) 

print('\nKuinka monta prosenttia miehistä ja naisista on käyttäneet työterveys, liikunta ja lomaosake-palveluita?')
print(df.pivot_table(['liikunta', 'lomaosake', 'työterveys'], index=['sukupuoli'], margins=False).applymap("{:.1%}".format))