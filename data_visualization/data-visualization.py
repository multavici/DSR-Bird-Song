from data_preparation import orders_df

import plotly as py
import cufflinks as cf


df = orders_df()
print(df)
df = df.set_index('bird_order')
print(df)
fig = df.iplot(kind='bar', title='Recordings per order', xTitle='Orders', yTitle='recordings', asFigure=True)

div_str = py.offline.plot(fig, output_type='div', include_plotlyjs=False)
print(div_str)