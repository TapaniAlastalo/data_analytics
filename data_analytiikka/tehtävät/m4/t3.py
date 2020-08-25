import numpy as np
import pandas as pd
import math

def dotDistanceFromLine(x, x0, y0):
    jx1 = x['x1']
    jy1 = x['y1']
    jx2 = x['x2']
    jy2 = x['y2']
    Ax = x0 - jx1 #vektorin A x-komponentti
    Ay = y0 - jy1 #vektorin A y-komponentti
    Bx = jx2 - jx1 #vektorin B x-komponentti
    By = jy2 - jy1 #vektorin B y-komponentti
    t = (Ax * Bx + Ay * By)/ (Bx * Bx + By * By) #pistetulojen suhde
    if (t < 0):
        t = 0 #jos piste ei osu janalle
    elif (t > 1):
        t = 1 #jos piste ei osu janalle
    lx = jx1 + t * Bx #lähimmän pisteen x
    ly = jy1 + t * By #lähimmän pisteen y
    dx = x0 - lx #pisteen ja lähimmän pisteen delta-x
    dy = y0 - ly #pisteen ja lähimmän pisteen delta-y
    return math.sqrt(dx * dx + dy * dy) #neliöjuuri dx^2+dy^2 -lausekkeesta

df = pd.read_excel('http://www.pekkavaris.net/pisteet1.xlsx', sheet_name='Sheet1', names=['x1', 'y1'], skiprows=6)

df['x2'] = df['x1'].shift(-1).fillna(0).astype(int)
df['y2'] = df['y1'].shift(-1).fillna(0).astype(int)
df.drop([136], inplace=True)

a = {'x0': -412, 'y0': -1832}
b = {'x0': -3012, 'y0': -1678}
c = {'x0': -3500, 'y0': -3450}

df['a-etäisyys'] = df.apply(dotDistanceFromLine, x0 = a['x0'], y0 = a['y0'], axis=1).round(3)
df['b-etäisyys'] = df.apply(dotDistanceFromLine, x0 = b['x0'], y0 = b['y0'], axis=1).round(3)
df['c-etäisyys'] = df.apply(dotDistanceFromLine, x0 = c['x0'], y0 = c['y0'], axis=1).round(3)

print(df)

aMin = df['a-etäisyys'].min()
print(aMin)
bMin = df['b-etäisyys'].min()
print(bMin)
cMin = df['c-etäisyys'].min()
print(cMin)