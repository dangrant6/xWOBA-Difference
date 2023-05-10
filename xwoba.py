import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

runners_on = pd.read_csv('2022 Runners.csv')
runners_off = pd.read_csv('2022 No Runners.csv')

newdata = pd.merge(runners_on, runners_off, on='player_name', suffixes=('_x','_y'))
newdata['xwobadifference'] = newdata['xwoba_x'] - newdata['xwoba_y']

# only using pitchers with more than 500 pitches
filtered_data = newdata[(newdata['total_pitches_x'] > 500) & (newdata['total_pitches_y'] > 500)]
X = filtered_data[['total_pitches_x', 'total_pitches_y']]
y = filtered_data['xwobadifference']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

new_pitches_x = [600, 550, 700]
new_pitches_y = [500, 600, 450]
new_data = pd.DataFrame({'total_pitches_x': new_pitches_x, 'total_pitches_y': new_pitches_y})
predictions = model.predict(new_data)

rmse = mean_squared_error(y_test, y_pred, squared=False)
for i in range(len(predictions)):
    print("Prediction", i+1, ":", predictions[i])
print("Root Mean Squared Error (RMSE):", rmse)

# list of top pitchers
sorted_data = filtered_data.sort_values(by='xwobadifference', ascending=False)
top_pitchers = sorted_data.head(10)['player_name']
print("Top 10 pitchers based on xwobadifference:")
for pitcher in top_pitchers:
    print(pitcher)

