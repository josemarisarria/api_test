import pandas as pd
from flask import Flask, jsonify,request
from flask_cors import CORS
import pickle
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
cors = CORS(app)


@app.route('/')
def home():
    return jsonify({'message': 'ONGI ETORRI housing API-ra!'})


#1 PREDICCION DE PRECIOS

# Cargar la base de datos en un DataFrame
model = pd.read_pickle('model.pkl')

@app.route('/predict/', methods=['POST'])
def predict():
    data = request.get_json()

    surface = int(data['surface'])
    bedrooms = int(data['bedrooms'])
    restrooms = int(data['restrooms'])

    input_data = [[surface, bedrooms, restrooms]]
    prediction = model.predict(input_data)

    return jsonify({'prediction': float(prediction[0])})


@app.route('/v1/predict/', methods=['GET'])
def predict_2():
    model = pickle.load(open('model.pkl','rb'))  ## rb = read binary,...hace falta poner eso

    surface = request.args.get('surface', None)
    bedrooms = request.args.get('bedrooms', None)
    restrooms = request.args.get('restrooms', None)


    if surface is None or bedrooms is None or restrooms is None:
        return "Missing args, the input values are needed to predict"
    else:
        prediction = model.predict([[int(surface), int(bedrooms), int(restrooms)]])
        return jsonify({'predictions' : float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
 

    '''if surface is None or bedrooms is None or restrooms is None:
        return "FALTAN VALORES, por favor introduzca los parámetros surface, bedrooms y restrooms"
    else:
        prediction = model.predict([[int(surface), int(bedrooms), int(restrooms)]])
        return print("LA PREDICCION DEL PRECIO DE LA VIVIENDA QUE BUSCA ES: " , prediction)'''


