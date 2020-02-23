import pandas as pd

df = pd.read_csv('https://student.labranet.jamk.fi/~varpe/datananal2k2020/kerta4/kysely.csv', sep = ',', decimal='.')
# korvataan puuttuvat arvot nollilla
df.fillna(0, inplace=True)
print(df)

print('lukumäärät siten, että riveillä perhesuhteet, sarakkeissa tyytyväisyys johtoon (lukumäärät, eli kuinka monta perheellistä on vastannut "tyytyväisyys johtoon"-kohtaan ykkösen jne')
print(df.pivot_table(['TyytJohto'], index=['perhetilanne']))    #, columns=['TyytJohto'], aggfunc=min, margins=False))

print('prosenttiosuudet siten, että riveillä koulutus tekstinä (esim peruskoulu), sarakkeissa tyytyväisyys työtehtäviin ( prosenttiosuudet 1 desimaalilla riveittäin eli esim montako % vain peruskoulun käyneistä vastaa 1 jne.)')

print('keskiarvot tyytyväisyyksistä johtoon ja työtehtäviin siten, että riveillä on ikäluokka 20-29, 30-39, ... 60-69 ja (eri sarakkeissa) tyytyväisyydet johtoon ja työtehtäviin')


print('Kuinka monta prosenttia miehistä ja naisista on käyttäneet työterveys, liikunta ja lomaosake-palveluita?')
#print(pd.crosstab(df['sukupuoli'],  [df['liikunta'], df['lomaosake'], df['työterveys']], normalize = 'index').applymap("{:.1%}".format))
#print(pd.crosstab(df['sukupuoli'],  df['liikunta', 'lomaosake', 'työterveys'], normalize = 'index').applymap("{:.1%}".format))

print(df.pivot_table(['liikunta', 'lomaosake', 'työterveys'], index=['sukupuoli'], margins=False).applymap("{:.1%}".format))