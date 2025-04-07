from flask import Flask, render_template, request
import pickle
import pandas as pd
import statsmodels.api as sm

app = Flask(__name__)

# Cargar el modelo guardado
with open('modelo_diabetes.pkl', 'rb') as archivo:
    modelo = pickle.load(archivo)

# Función para hacer predicciones con nuevos datos
def predecir_diabetes(edad, bmi):
    nuevos_datos = pd.DataFrame({'Age': [edad], 'BMI': [bmi]})
    nuevos_datos_sm = sm.add_constant(nuevos_datos, has_constant='add')
    prediccion = modelo.predict(nuevos_datos_sm)
    resultado = 1 if prediccion[0] > 0.5 else 0
    return resultado, prediccion[0]

# Ruta principal para mostrar el formulario
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para manejar la predicción
@app.route('/predecir', methods=['POST'])
def predecir():
    # Obtener los datos del formulario
    edad = float(request.form['edad'])
    bmi = float(request.form['bmi'])

    # Hacer la predicción
    resultado, probabilidad = predecir_diabetes(edad, bmi)

    # Determinar el mensaje según la predicción
    if resultado == 1:
        mensaje = f"Probabilidad de tener diabetes: {probabilidad:.2f}. Deberías pedir cita al médico."
    else:
        mensaje = f"Probabilidad de tener diabetes: {probabilidad:.2f}. Relájate, cómete una pizza."

    # Renderizar la página con el resultado
    return render_template('index.html', mensaje=mensaje)

if __name__ == '__main__':
    app.run(debug=True)
