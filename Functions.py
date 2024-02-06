import pandas as pd
import sqlite3
from sklearn.preprocessing import MinMaxScaler  # Or any other scaler you used
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
def select_data(id_number = 20, lags=1, Normalize=False, plotting=False, CHANTYPE=2, drop_cols = ['STATE', 'hour', 'minute', 'second']):
    param_info = ('lags: ' + str(lags) + ', ', 'Normalize: ' + str(Normalize) + ', ', 'plotting: ' + str(plotting) + ', ',
     'CHANTYPE: ' + str(CHANTYPE) + ', ', 'drop_cols: ' + str(drop_cols))
    # Step 1: Establish a connection to the SQLite database (creates a new file if not exists)
    connection = sqlite3.connect("./data/UtilitySmartData_Anomaly.db")

    # Step 2: Create a cursor
    cursor = connection.cursor()

    # Step 3: Execute SQL command to get a list of tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    # Step 4: Fetch all table names
    table_names = cursor.fetchall()

    id = table_names[id_number][0]

    # Step 3: Execute SQL command to fetch data from the "users" table
    df = pd.read_sql_query("SELECT * FROM " + "'" + id + "'", connection)
    if CHANTYPE==1:
        df = df[df['CHANTYPE']==CHANTYPE]
        df = df.drop('CHANTYPE', axis=1)
    elif CHANTYPE==2:
        df = df[df['CHANTYPE']==CHANTYPE]
        df = df.drop('CHANTYPE', axis=1)

    df.set_index('READTS', inplace=True)

    # Replace zeros with NaN
    df['VAL'].replace(0, pd.NA, inplace=True)

    # Use bfill (backward fill) to fill NaN values with the last non-NaN value
    df['VAL'].fillna(method='bfill', inplace=True)

    # df['lag_' + str(lags) + '_VAL'] = df['VAL'].shift(lags)

    if lags>0:
        for lag in range(1, lags+1):
            df['lag_' + str(lag) + '_VAL'] = df['VAL'].shift(lag)

    df.dropna(inplace=True)

    if len(df['year'].unique())>1:
        train_data = df[df['year']==2021]
        test_data = df[df['year']==2022]
    else:
        # Step 2: Calculate the number of rows for 80% of the total rows
        num_rows = int(len(df) * 0.8)
        # Step 3: Select the first 80% records as training data
        train_data = df[0:num_rows]
        # Step 4: Select the last 20% records as test data
        test_data = df[num_rows:df.shape[0]]

    # columns_to_drop = ['year', 'READTS']  # Replace with the names of the columns you want to drop
    train_data = train_data.drop(columns=['year'])
    test_data = test_data.drop(columns=['year'])

    # Separate the features and the target column ('VAL') in train_data
    X_train = train_data.drop(columns=['VAL'])
    y_train = train_data['VAL']

    # Separate the features and the target column ('VAL') in test_data
    X_test = test_data.drop(columns=['VAL'])
    y_test = test_data['VAL']

    X_columns = X_test.columns
    train_index = X_train.index
    test_index = X_test.index

    # Normalize the features to have given range on the training set, e.g. between zero and one.
    if Normalize==True:
        # scaler = StandardScaler()
        scaler = MinMaxScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(data=X_train, columns=X_columns, index=train_index)
    X_test = pd.DataFrame(data=X_test, columns=X_columns, index=test_index)

    if len(drop_cols)>0:
        X_train = X_train.drop(columns= drop_cols)
        X_test = X_test.drop(columns= drop_cols)

    # Step 5: Close the connection
    connection.close()

    # import plotly.express as px
    #
    # # Step 2: Create a line plot using Plotly Express
    # fig = px.line(df, x='READTS', y='VAL', title='Line Plot for "VAL" Column')
    #
    # # Step 3: Show the plot
    # fig.show()

    if plotting==True:
        # Ensure the index is of datetime type
        # df.index = pd.to_datetime(df.index)

        # Count total columns for subplots excluding 'VAL' column
        n_total_cols = len(drop_cols) - 1  # Excluding 'VAL'

        # Define the figure size
        plt.figure(figsize=(20, 25))

        # Loop through all columns except 'VAL' to plot either violinplot or barplot
        for i, col in enumerate(drop_cols):
            if col != 'VAL':
                # Calculate the grid position
                row_index = i // (n_total_cols // 2)
                col_index = i % (n_total_cols // 2)

                ax = plt.subplot2grid((4, n_total_cols // 2), (row_index, col_index), rowspan=1, colspan=1)

                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.tick_params(axis='both', which='minor', labelsize=8)

                # If column has 5 or fewer unique values, use bar plot
                if df[col].nunique() <= 5:
                    value_counts = df[col].value_counts()
                    ax.bar(value_counts.index, value_counts.values)
                    ax.set_title(f"{col}")
                    ax.set_xticklabels(value_counts.index, rotation=45)

                # If column has more than 5 unique values, use violin plot
                else:
                    ax.violinplot(df[col])
                    ax.set_title(f"{col}")

        # Line plot for expected values, in the last subplot position
        ax2 = plt.subplot2grid((4, n_total_cols // 2), (2, 0), rowspan=2, colspan=n_total_cols // 2)
        ax2.plot(df.index, df.VAL, label='real')
        # Format the x-axis to only show year and month
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        ax2.tick_params(axis='both', which='major', labelsize=10)
        ax2.tick_params(axis='both', which='minor', labelsize=8)
        ax2.legend()
        ax2.set_title("Line plot of VAL")

        # Adjust spacing between subplots for better visibility
        plt.subplots_adjust(hspace=0.4, wspace = 0.5)

        plt.tight_layout()
        plt.show()
    result = {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test, 'id':id, 'param_info':param_info}
    return result