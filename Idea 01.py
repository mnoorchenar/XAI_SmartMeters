import Functions as myfuc

df = myfuc.select_data(id_number = 20, lags=1, Normalize=True, plotting=False, CHANTYPE=2, drop_cols = ['STATE', 'hour', 'minute', 'second'])
#['month', 'day', 'weekday', 'is_weekend', 'time_of_day', 'Temperature', 'Dew_Point_Temperature', 'Relative_Humidity', 'Precipitation_Amount', 'Wind_Direction', 'Wind_Speed',
# 'Visibility', 'Station_Pressure', 'Anomaly', 'lag_1_VAL']

# Assuming your data is in a dictionary named 'df'
X_train = df['X_train']
y_train = df['y_train']
X_test = df['X_test']
y_test = df['y_test']
