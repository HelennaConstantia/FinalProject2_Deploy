from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
app=Flask(__name__)
model=pickle.load(open('model_LR2.pkl','rb'))
scaler=StandardScaler()
@app.route("/")
def form():
    return render_template("LR_Predict.html")
@app.route("/predict",methods=['POST'])
def predict():
    rainfall=float(request.form['rainfall'])
    wind=float(request.form['wind'])
    humidity=float(request.form['humidity'])
    pressure=float(request.form['pressure'])
    rain=request.form['rain']
    x_input=[rainfall,wind,humidity,pressure]
    if rain=='yes':
        x_input.append(1)
    elif rain=='no':
        x_input.append(0)
    x_input=scaler.fit_transform([x_input])
 
    prediction=model.predict(x_input)
    output={0:'tidak akan hujan', 1:'akan hujan'}
  
    return render_template('LR_Predict.html',prediction_text="Diprediksi besok {}".format(output[prediction[0]]))

if __name__ == "__main__":
    app.run(debug=True)


