'''
 # @ Create Time: 2023-12-10 14:56:12.646697
'''

from dash import Dash, html, dcc, dash_table
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

# Download the newest data
urlLocation = 'https://aqicn.org/data-platform/covid19/report/39374-7694ec07/'
csvFile = urlDownload(urlLocation)

# Define the columns to load
meta_cols = ['Date', 'Country', 'City', 'Specie']
main_column = 'median' # 'count', 'min', 'max', 'median', 'variance'
selected_cols = meta_cols + [main_column]

# Read the newest data file and skip the first 4 lines
DF = pd.read_csv(csvFile, skiprows=4, usecols=selected_cols, nrows=50)

app = Dash(__name__, title="DashTest")

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# App layout
app.layout = html.Div([
    html.Div(children='My First App with Data, Graph, and Controls'),
    html.Hr(),
    dash_table.DataTable(data=DF.to_dict('records'), page_size=6)
])

if __name__ == '__main__':
    app.run_server(debug=True)
