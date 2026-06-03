import pandas as pd

PATH = '/home/edgar/Dev/projects/semantic-cluster/data/samples/maestro_es.csv'

df = pd.read_csv(PATH)

# print(df.describe())
# print(df['ubicacion'].unique())
# print(df['lugar_especifico'].unique())
# riviera_maya = df[df['ubicacion'] == 'Rivera Maya']
riviera_nayarit = df[df['ubicacion'] == 'Riviera Nayarit']
la_paz = df[df['ubicacion'] == 'La Paz']
puerto_vallarta = df[df['ubicacion'] == 'Puerto Vallarta']
huatulco = df[df['ubicacion'] == 'Huatulco']

# riviera_maya.to_csv('data/samples/riviera_maya.csv', index=False)
riviera_nayarit.to_csv('data/samples/riviera_nayarit.csv', index=False)
la_paz.to_csv('data/samples/la_paz.csv', index=False)
puerto_vallarta.to_csv('data/samples/puerto_vallarta.csv', index=False)
huatulco.to_csv('data/samples/huatulco.csv', index=False)

