# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dask.dataframe as dd
import multiprocessing as mp
import prophet
from prophet.diagnostics import performance_metrics, cross_validation
from prophet.plot import plot_plotly, plot_components_plotly

# Load required Python packages
import os, re, requests, logging
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import gc

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

# Create lists of year and quarter names
yNames = [str(i) for i in range(2020, 2022)]
qNames = ["Q" + str(i) for i in range(1, 5)]

# Create a data frame with the url locations and year/quarter combinations
DF = pd.DataFrame(list(product(yNames, qNames)),columns=['yNames', 'qNames'])
DF.insert(loc=0, column='urlLocation', value=urlLocation)

# Combine url location and year/quarter combinations into a single column
DF = pd.DataFrame({'urlLocations': DF.agg(''.join, axis=1)})

# Download legacy data (in parallel)
DDF = dd.from_pandas(DF.iloc[:2], npartitions=mp.cpu_count())
csvFiles = DDF.apply(lambda x : urlDownload(x[0]), axis=1, meta=pd.Series(dtype="str")).compute(scheduler='threads')
collected = gc.collect()
print(f'COLLECTED: {collected}')
print('DOWNLOADED FIRST 2')

DDF = dd.from_pandas(DF.iloc[2:4], npartitions=mp.cpu_count())
csvFiles = DDF.apply(lambda x : urlDownload(x[0]), axis=1, meta=pd.Series(dtype="str")).compute(scheduler='threads')
collected = gc.collect()
print(f'COLLECTED: {collected}')
print('DOWNLOADED SECOND 2')

DDF = dd.from_pandas(DF.iloc[4:6], npartitions=mp.cpu_count())
csvFiles = DDF.apply(lambda x : urlDownload(x[0]), axis=1, meta=pd.Series(dtype="str")).compute(scheduler='threads')
collected = gc.collect()
print(f'COLLECTED: {collected}')
print('DOWNLOADED THIRD 2')


# Define the columns to load
meta_cols = ['Date', 'Country', 'City', 'Specie']
main_column = 'median' # 'count', 'min', 'max', 'median', 'variance'
selected_cols = meta_cols + [main_column]

# Read the newest data file and skip the first 4 lines
DF = pd.read_csv(csvFile, skiprows=4, usecols=selected_cols)

# Leave Italy data, rename main column to Value
selectIT = DF['Country'].isin(['IT'])
newTable = DF[selectIT].rename(columns={main_column: 'Value'})

# Read legacy data files (sequentially)
fileNamesQ = [f for f in os.listdir('.') if re.match(r'^.*Q\d.csv$', f)]
DF = pd.concat((pd.read_csv(f, skiprows=4, usecols=selected_cols) for f in fileNamesQ), ignore_index=True)
selectIT = DF['Country'].isin(['IT'])
oldTable = DF[selectIT].rename(columns={main_column: 'Value'})

# Append old (2018-2021) and new (2022-2023) data tables, sort, remove duplicates
DF = pd.concat([oldTable, newTable])
dataTableIT = DF.sort_values(by=['Country', 'City', 'Date']).drop_duplicates()

# Calculate the proportion of each Species in the data table
all_vars = 100 * pd.Series(dataTableIT.Specie).value_counts() / len(dataTableIT)

# Drop the variables that are not needed
drop_weat = ['pressure', 'wind-speed', 'wind-gust', 'wind speed', 'wind gust', 'dew', 'precipitation', 'temperature', 'humidity']
drop_poll = ['wd', 'aqi', 'uvi', 'pm1', 'neph', 'mepaqi']
keep_vars = set(all_vars.index) - set(drop_weat + drop_poll)

# Create a new data table with the info on kept variables
new_data_table = pd.DataFrame([all_vars[list(keep_vars)].sort_values(ascending=False)])
new_data_table.style.hide(axis="index")

# 2021-10-03 Barcelona fix
dataTableEU = dataTableIT.groupby(['Date', 'Country', 'City', 'Specie'])[['Value']].mean().reset_index()

# Create pivot table, calculate API for each row, drop rows with missing API values
dataTableAPI = dataTableIT.pivot_table(index=['Date', 'Country', 'City'], columns='Specie', values='Value').reset_index()
dataTableAPI["API"] = np.maximum.reduce(dataTableAPI[['pm10', 'pm25', 'no2', 'co', 'o3', 'so2']].values, axis=1)
dataTableAPI = dataTableAPI.dropna(subset=["API"])

cities = dataTableAPI['City'].unique()

app = Dash(__name__, title="DashTest")

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# App layout
app.layout = html.Div([
    html.H1(children='Italy API analysis'),
    html.Hr(),
    html.Br(),
    dcc.Dropdown(cities, value=cities[0], id='controls-city-dropdown'),
    html.Br(),
        dcc.Tabs([
            dcc.Tab(label='API statistics', children=[
                html.Br(),
                html.Label('Descriptive statistics'),
                dash_table.DataTable(data=[], page_size=6, id='city-table'),
                html.Br(),
                dcc.Graph(figure={}, id='city-api-history'),
                html.Br(),
                dcc.Graph(figure={}, id='city-weekday-matrix'),
                html.Br(),
                dcc.Graph(figure={}, id='city-year-matrix'),
            ]),
             dcc.Tab(label='Forecasts', children=[
                # dcc.Graph(figure={}, id='city-forecast', responsive=True),
                # html.Br(),
                # html.Label('Backtesting errors'),
                # dash_table.DataTable(data=[], id='city-forecast-errors'),
            ])
        ])
])

@callback(
    Output(component_id='city-table', component_property='data'),
    Input(component_id='controls-city-dropdown', component_property='value')
)
def update_table(city_chosen):
    cityData = dataTableAPI[(dataTableAPI['City']==city_chosen) & (dataTableAPI['Date']>='2019-01-01')][["City","Date", "API"]]
    groupedCityData = cityData.groupby(pd.to_datetime(cityData['Date']).dt.year)
    yearlyData = pd.DataFrame(groupedCityData.describe())
    yearlyData.insert(0, ('API', 'Date'), list(groupedCityData.groups.keys()))
    return yearlyData['API'].to_dict('records')


@callback(
    Output(component_id='city-api-history', component_property='figure'),
    Input(component_id='controls-city-dropdown', component_property='value')
)
def update_api_historical(city_chosen):
    data = dataTableAPI[(dataTableAPI['City']==city_chosen) & (dataTableAPI['Date']>='2019-01-01')][["City","Date", "API"]]
    fig = px.line(data, x='Date', y='API', title=f'Historical API data for {city_chosen}')
    return fig

@callback(
    Output(component_id='city-weekday-matrix', component_property='figure'),
    Input(component_id='controls-city-dropdown', component_property='value')
)
def filter_heatmap(city_chosen):
    cityData = dataTableAPI[(dataTableAPI['City']==city_chosen) & (dataTableAPI['Date']>='2019-01-01')][["Date", "API"]]
    groupedData = pd.DataFrame(data=cityData.groupby(pd.to_datetime(cityData['Date']).dt.weekday)['API'].mean())
    fig = px.imshow(groupedData.to_numpy().transpose(),
                    labels=dict(x='Day of the Week', y='API'),
                    color_continuous_scale=[[0, 'green'], [0.5, 'yellow'], [1.0, 'red']],
                    y=[''],
                    x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    return fig

@callback(
    Output(component_id='city-year-matrix', component_property='figure'),
    Input(component_id='controls-city-dropdown', component_property='value')
)
def filter_year_heatmap(city_chosen):
    cityData = dataTableAPI[(dataTableAPI['City']==city_chosen) & (dataTableAPI['Date']>='2019-01-01')][["Date", "API"]]
    groupedCityData = cityData.groupby(pd.to_datetime(cityData['Date']).dt.year)['API']
    groupedData = pd.DataFrame(data=groupedCityData.mean())
    fig = px.imshow(groupedData.to_numpy().transpose(),
                    labels=dict(x='Year', y='API'),
                    color_continuous_scale=[[0, 'green'], [0.5, 'yellow'], [1.0, 'red']],
                    y=[''],
                    x=list(map(str, list(groupedCityData.groups.keys()))))
    return fig

# @callback(
#     Output(component_id='city-forecast', component_property='figure'),
#     Input(component_id='controls-city-dropdown', component_property='value')
# )
# def update_forecast(city_chosen):
#     fullData = dataTableAPI[(dataTableAPI['City']==city_chosen)][['Date', 'API', 'pm25', 'pm10', 'no2', 'co', 'o3', 'so2', 'temperature', 'humidity']]
#     fullData = fullData.reset_index()
#     fullData = fullData.dropna(subset=["temperature", "humidity"])
#     temperatureFullData = fullData
#     humidityFullData = fullData
#     fullData = fullData.rename(columns={'Date':'ds', 'API':'y'})
#     temperatureFullData = temperatureFullData.rename(columns={'Date':'ds', 'temperature':'y'})
#     humidityFullData = humidityFullData.rename(columns={'Date':'ds', 'humidity':'y'})
#     forecastingRange = 28

#     #Humidity
#     humidityModel = prophet.Prophet(weekly_seasonality=True, daily_seasonality=True)
#     humidityModel.fit(humidityFullData)
#     humidityFuture = humidityModel.make_future_dataframe(periods=forecastingRange, freq = 'd')
#     humidityForecast = humidityModel.predict(humidityFuture)

#     #Temperature
#     temperatureModel = prophet.Prophet(weekly_seasonality=True, daily_seasonality=True)
#     temperatureModel.fit(temperatureFullData)
#     temperatureFuture = temperatureModel.make_future_dataframe(periods=forecastingRange, freq = 'd')
#     temperatureForecast = temperatureModel.predict(temperatureFuture)

#     #API - multi
#     APIModel = prophet.Prophet(weekly_seasonality=True, daily_seasonality=True)
#     APIModel.add_regressor('temperature')
#     APIModel.add_regressor('humidity')
#     APIModel.fit(fullData)
#     APIFuture = APIModel.make_future_dataframe(periods=forecastingRange, freq = 'd')
#     APIFuture['temperature']=temperatureForecast['yhat']
#     APIFuture['humidity']=humidityForecast['yhat']
#     APIForecast = APIModel.predict(APIFuture)
#     fig = plot_plotly(APIModel, APIForecast)
#     fig.update_layout(title_text=f'Forecast for {city_chosen} for next 28 days', title_font_size=16)
#     fig.update_xaxes(title_text='Date')
#     fig.update_yaxes(title_text='API')
#     return fig

# @callback(
#     Output(component_id='city-forecast-errors', component_property='data'),
#     Input(component_id='controls-city-dropdown', component_property='value')
# )
# def update_forecast_errors(city_chosen):
#     fullData = dataTableAPI[(dataTableAPI['City']==city_chosen)][['Date', 'API', 'pm25', 'pm10', 'no2', 'co', 'o3', 'so2', 'temperature', 'humidity']]
#     fullData = fullData.reset_index()
#     fullData = fullData.dropna(subset=["temperature", "humidity"])
#     fullData = fullData.rename(columns={'Date':'ds', 'API':'y'})
#     forecastingRange = 28
#     minDate = datetime.strptime(fullData['ds'].agg(['min', 'max'])[0], '%Y-%m-%d').date()
#     maxDate = datetime.strptime(fullData['ds'].agg(['min', 'max'])[1], '%Y-%m-%d').date()
#     trainingDays = (maxDate - minDate).days - forecastingRange

#     #API - multi
#     APIModel = prophet.Prophet(weekly_seasonality=True, daily_seasonality=True)
#     APIModel.add_regressor('temperature')
#     APIModel.add_regressor('humidity')
#     APIModel.fit(fullData)

#     df_cv = cross_validation(APIModel, initial=f'{trainingDays} days', period=f'{forecastingRange} days', horizon = f'{forecastingRange} days')
#     df_p = performance_metrics(df_cv)
#     return df_p.tail(1)[['mse','rmse','mae','mape','mdape','smape']].to_dict('records')


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
