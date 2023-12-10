'''
 # @ Create Time: 2023-12-10 14:56:12.646697
'''

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

# Load required Python packages
import os, requests

# Function to download remote file to the disk
def urlDownload(urlLink):
  with requests.get(urlLink, stream=True) as r:
    fileSize = int(r.headers.get('Content-Length'))
    fileName = r.headers.get('Content-Disposition').split("filename=")[1]
    if not os.path.exists(fileName) or os.path.getsize(fileName) != fileSize:
      block_size = 1024
      with open(fileName, 'wb') as file:
        for data in r.iter_content(block_size):
          file.write(data)
    return fileName

# Download the newest data
urlLocation = 'https://aqicn.org/data-platform/covid19/report/39374-7694ec07/'
csvFile = urlDownload(urlLocation)

app = Dash(__name__, title="DashTest")

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
