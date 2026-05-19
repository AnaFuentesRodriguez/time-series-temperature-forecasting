# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 11:38:07 2025

@author: Usuario
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # for 1 block validation
from scipy.optimize import curve_fit
from scipy.stats import shapiro # normality test
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # AutoCorrelation Function  and Partial ACF
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose


# Cargar datos
data = pd.read_csv('oikolab.csv')

#Primeras líneas del conjunto
print('Primeras instancias:')
print(data.head())

#Información general
print('\nInformación general:')
print(data.info())

#Valores nulos en cada columna
print('\nValores nulos:')
print(data.isnull().sum())

# Convertir la columna DateTime a tipo fecha
data['DateTime'] = pd.to_datetime(data['DateTime'])

# Establecer la fecha como índice
data.set_index('DateTime', inplace=True)

# Graficar la temperatura a lo largo del tiempo
plt.figure(figsize=(12,5))
plt.plot(data['Temperature (ºC)'], label='Temperatura (ºC)')
plt.xlabel('Fecha')
plt.ylabel('Temperatura (ºC)')
plt.title('Evolución de la Temperatura')
plt.legend()
plt.grid()
plt.show()


# Resamplear a nivel mensual y calcular la media de temperatura
temp_monthly = data['Temperature (ºC)'].resample('ME').mean()
#Convertir a DataFrame
df = temp_monthly.to_frame(name='Temperatura_media')

# Graficar la temperatura a lo largo del tiempo
plt.figure(figsize=(12,5))
plt.plot(df['Temperatura_media'], label='Temperatura (ºC)')
plt.xlabel('Fecha')
plt.ylabel('Temperatura (ºC)')
plt.title('Evolución de la Temperatura Media')
plt.legend()
plt.grid()
plt.show()


# Función para verificar estacionariedad
def check_stationarity(columna, filter_size):
    # Calcular la media y desviación estándar móvil
    rolling_mean = df[columna].rolling(window=filter_size).mean()
    rolling_std = df[columna].rolling(window=filter_size).std()
    
    # Graficar la serie original con la media y desviación móvil
    plt.figure(figsize=(12, 6))
    plt.plot(df[columna], label='Serie Original', color='blue')
    plt.plot(rolling_mean, label='Media Móvil', color='red', linestyle='dashed')
    plt.plot(rolling_std, label='Desviación Estándar Móvil', color='black', linestyle='dotted')
    plt.title('Serie Temporal con Media y Desviación Móvil')
    plt.xlabel('Año')
    plt.ylabel('Temperatura (ºC)')
    plt.legend()
    plt.grid()
    plt.show()
    
    # Graficar ACF y PACF
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    plot_acf(df[columna].dropna(), ax=axes[0])
    plot_pacf(df[columna].dropna(), ax=axes[1])
    plt.show()
    
    # Test de Dickey-Fuller
    print('Resultados del Test de Dickey-Fuller:')
    dftest = adfuller(df[columna].dropna(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Estadístico ADF', 'p-valor', '#Lags Usados', 'Número de Observaciones'])
    for key, value in dftest[4].items():
        dfoutput[f'Valor Crítico ({key})'] = value
    print(dfoutput, '\n')
    
    if dftest[1] < 0.05:
        print("Conclusión: La serie SÍ es estacionaria.")
    else:
        print("Conclusión: La serie NO es estacionaria.")

# Aplicar la función a la serie temporal
check_stationarity('Temperatura_media', filter_size=12)


#Diferenciar
df['Temperatura_diferenciada'] = df['Temperatura_media'].diff().dropna()

# Graficar la serie diferenciada
plt.figure(figsize=(12, 6))
plt.plot(df['Temperatura_diferenciada'], label='Temperatura Diferenciada (d=1)', color='r')
plt.title('Serie Temporal de Temperatura Diferenciada (d=1)')
plt.xlabel('Año')
plt.ylabel('Diferencia de Temperatura (ºC)')
plt.legend()
plt.grid()
plt.show()


# Aplicar la función a la serie temporal diferenciada
check_stationarity('Temperatura_diferenciada', filter_size=12)


# Aplicar la descomposición de la serie
columna = 'Temperatura_media'
decomposition = seasonal_decompose(df[columna], model='additive', period=12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Función para graficar la descomposición
def plot_decomposition(series, trend, seasonal, residual):
    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(series, label='Serie Original', color='blue')
    plt.legend()
    plt.grid()
    plt.subplot(412)
    plt.plot(trend, label='Tendencia', color='red')
    plt.legend()
    plt.grid()
    plt.subplot(413)
    plt.plot(seasonal, label='Estacionalidad', color='green')
    plt.legend()
    plt.grid()
    plt.subplot(414)
    plt.plot(residual, label='Residuo', color='black')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

# Llamar a la función con la serie y sus componentes
plot_decomposition(df[columna], trend, seasonal, residual)



#Eliminar estacionalidad
x_ts= df['Temperatura_media'].to_numpy()
t= np.arange(len(x_ts))
def RemoveSeasonality(data, period, show=True):

    # Season model
    Season= np.zeros(period)
    for i in range(period):
        Season[i]= np.mean(data[i::period]) # Take all time series values from i, in steps of period


    # Get number of seasons in the data and tile season model
    NumSeasons= int(np.ceil(len(x_ts)/len(Season)))
    TiledSeason= np.tile(Season, NumSeasons) # Tile seasonal model on time series
    
    # Skew tiled seasonality to data length
    TiledSeason= TiledSeason[:len(x_ts)]

    if show:
        plt.figure()
        plt.plot(t, data)
        plt.plot(t, TiledSeason)
        plt.legend(['Time Series', 'Seasonality model'])

    # Remove seasonality
    s_data= data - TiledSeason
    return Season, s_data

S= 12 # Seasonality period

# Now remove seasonality, and get season model in the process
season_model, s_data= RemoveSeasonality(data= df['Temperatura_media'].to_numpy(), period=S)
df['S_Temperatura_media']= s_data

# Check stationarity again
check_stationarity('S_Temperatura_media', filter_size=12)


# Create a grid of possible values for p and q
possible_p= [0, 1, 2, 3]
possible_q= [0, 1, 2, 3]
d= 0 # No se ha diferenciado para que sea estacionaria

parameter_grid= []
for p in possible_p:
    for q in possible_q:
        if p!= 0 or q != 0:
            parameter_grid.append((p,q))

from collections import namedtuple

result=namedtuple('result', ['model', 'p', 'q', 'trainMSE', 'testMSE', 'AIC'])

# Get data
x_ts= df['S_Temperatura_media'].dropna().to_numpy()

# divide in train and test ( one block validation)
NTest= 12 # 12 values to predict

tr_data, ts_data= train_test_split(x_ts, test_size=NTest, shuffle=False)
t_tr= np.arange(len(tr_data))
t_ts= t_tr[-1]+np.arange(len(ts_data))+1


# Fit each ARIMA(p,d,q) model
all_results= []
df_results = pd.DataFrame(columns=['p', 'q', 'AIC', 'trMSE', 'tsMSE'])
for p,q in parameter_grid:
    
    # Get model
    model = ARIMA(tr_data, order=(p,d,q))
    
    # Fit model
    fitted_model= model.fit()
    
    # get AIC value
    AIC= fitted_model.aic
    
    # Calculate training MSE
    tr_residuals= tr_data-fitted_model.fittedvalues

    trainMSE= np.sum(tr_residuals**2)/len(tr_data)
    
    # Predict next NTest values with ARIMA
    predictions= fitted_model.get_prediction(start=len(tr_data), end=len(x_ts)-1).predicted_mean
    
    # Calculate test MSE
    ts_residuals= ts_data-predictions
    testMSE= np.sum(ts_residuals**2)/len(ts_data)
    
    all_results.append( result(model=fitted_model, p=p, q=q, trainMSE=trainMSE, testMSE=testMSE, AIC=AIC))
    df_results = pd.concat([df_results, pd.DataFrame([{'p': p, 'q': q, 'AIC': AIC, 'trMSE': trainMSE, 'tsMSE': testMSE}])], ignore_index=True)

all_aics= [experiment.AIC for experiment in all_results]
sorted_idx= np.argsort(all_aics)
bestExperiment= all_results[sorted_idx[0]]
bestAIC, bestP, bestQ= bestExperiment.AIC, bestExperiment.p, bestExperiment.q

print('Best AIC is {:.3f} for model ARIMA({}, {}, {})'.format(bestAIC, bestP, d, bestQ))
selected_model= bestExperiment.model

selected_model.plot_diagnostics()


predictions= selected_model.get_prediction(start=len(tr_data), end=len(x_ts)-1).predicted_mean

plt.figure()
plt.plot(t_tr, tr_data)
plt.plot(t_tr, selected_model.fittedvalues)
plt.plot(t_ts, ts_data)
plt.plot(t_ts, predictions)
plt.legend(['Training data', 'Training fitted data', 'test data', 'test predictions'])
plt.title('Entrenamiento y evaluación')
plt.show()


###########################
#### Undo seasonality model

# Tile seasonality
t= np.arange(-d, t_ts[-1]+1)+d
total_data= len(t)
NumSeasons= int(np.ceil(total_data/len(season_model)))
TiledSeason= np.tile(season_model, NumSeasons)[:total_data] # Tile seasonal model on time series

# Get seasonality for test
test_seasonality= TiledSeason[-len(t_ts):]

# Add seasonality to time series
predictions+= test_seasonality


########################## 
# Plot predictions vs test data
plt.figure()

plt.plot(t, df['Temperatura_media'].to_numpy())
plt.plot(t_ts+d, predictions)
plt.legend(['Real data', 'Predicted values'])
plt.title('Predicciones de los últimos valores')

val_MSE= np.mean((df['Temperatura_media'].to_numpy()[-NTest:]-predictions)**2)
print('The validation MSE in the true time series scale is: ', val_MSE)


#Predeecir los valores de junio a diciembre de 2021 (ambos incluidos
# Eliminar el último valor del dataset (que corresponde al 1 de junio de 2021)
df = df.drop(df.index[-1])  

n_predicciones = 7
#Predecir los meses de junio a diciembre 2021 con el modelo ARIMA
forecast_arima = selected_model.forecast(steps=n_predicciones)

#Obtener la componente estacional para esos mismos meses
forecast_season = season_model[-n_predicciones:]  # Ajusta el slicing según cómo tengas season_model

#Sumar ambas partes
forecast_final = forecast_arima + forecast_season

#Crear el índice para las predicciones (de junio a diciembre de 2021)
forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=n_predicciones, freq='ME')

#Convertir las predicciones a DataFrame
forecast_df = pd.DataFrame({'Fecha': forecast_index, 'Predicción': forecast_final})
forecast_df.set_index('Fecha', inplace=True)

# Graficar las predicciones junto con la serie original
plt.figure(figsize=(12, 6))
plt.plot(df['Temperatura_media'], label='Datos reales', color='blue')
plt.plot(forecast_df, label='Predicción', color='red', linestyle='dashed')
plt.xlabel('Fecha')
plt.ylabel('Temperatura (ºC)')
plt.title('Predicción de temperatura para los próximos meses')
plt.legend()
plt.grid()
plt.show()

print(forecast_df)





