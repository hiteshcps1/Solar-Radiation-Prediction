import flask
import pandas as pd
from joblib import dump, load



with open(f'linear.joblib', 'rb') as f:
    model = load(f)

with open(f'forest.joblib', 'rb') as f1:
    model1 = load(f1)


app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))

    if flask.request.method == 'POST':
        temp = flask.request.form['temp']
        Pressure = flask.request.form['Pressure']
        Humidity = flask.request.form['Humidity']
        wind = flask.request.form['wind']
        speed  = flask.request.form['speed']
        DayOfYear = flask.request.form['DayOfYear']
        TimeOfDay = flask.request.form['TimeOfDay']
        temp1 = flask.request.form['temp1']
        DayOfYear1 = flask.request.form['DayOfYear1']
        TimeOfDay1 = flask.request.form['TimeOfDay1']

        

        input_variables = pd.DataFrame([[temp,Pressure,Humidity,wind,speed,DayOfYear,TimeOfDay]],
                                       columns=['temp','Pressure','Humidity','wind','speed','DayOfYear','TimeOfDay'],
                                       dtype='float',
                                       index=['input'])
        input_variables1 = pd.DataFrame([[temp1,DayOfYear1,TimeOfDay1]],
                                       columns=['temp1','DayOfYear1','TimeOfDay1'],
                                       dtype='float',
                                       index=['input'])

        predictions = round(model.predict(input_variables)[0],2)
        print(predictions)
        predictions1 = round(model1.predict(input_variables1)[0],2)
        print(predictions1)

        return flask.render_template('main.html', original_input={'temp': temp, 'Pressure': Pressure, 'Humidity': Humidity, 'wind': wind, 'speed': speed, 'DayOfYear': DayOfYear, 'TimeOfDay': TimeOfDay, 'temp1': temp1, 'DayOfYear1': DayOfYear1, 'TimeOfDay1': TimeOfDay1},
                                     result=predictions,original_input1={'temp': temp, 'DayOfYear': DayOfYear, 'TimeOfDay': TimeOfDay, 'temp1': temp1, 'DayOfYear1': DayOfYear1, 'TimeOfDay1': TimeOfDay1},result1=predictions1)

        
if __name__ == '__main__':
    app.run(debug=True)

