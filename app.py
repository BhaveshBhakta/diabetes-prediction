from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

parkinsons_model = pickle.load(open("diabetes_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")  

@app.route("/predict_page", methods=["GET"])
def predict_page():
    return render_template("predict.html") 


@app.route("/predict", methods=["POST"])
def predict():
    result = None
    if request.method == 'POST':
       
        user_input = [
            request.form.get('Pregnancies'),
            request.form.get('Glucose'),
            request.form.get('BloodPressure'),
            request.form.get('SkinThickness'),
            request.form.get('Insulin'),
            request.form.get('BMI'),
            request.form.get('DiabetesPedigreeFunction'),
            request.form.get('Age')
        ]
        
        user_input = [float(x) if x else 0.0 for x in user_input]
     
        prediction = parkinsons_model.predict([user_input])
        result = "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"
    
    return render_template("predict.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)
