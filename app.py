# Import necessary libraries
from flask import Flask, request, render_template
import numpy as np
import pickle
from crop_recommendation import ModifiedRFE

# Load the trained model and preprocessing pipeline
model = pickle.load(open('model.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))
sc=pickle.load(open('standscaler.pkl','rb'))
pre_pro = pickle.load(open('preprocessing_pipeline.pkl', 'rb'))
# Create Flask app
app = Flask(__name__)

# Define routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Extract input data from form
    N = request.form['Nitrogen']
    P = request.form['Phosphorus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    hum = request.form['Humidity']
    ph = request.form['PH']
    rain = request.form['Rainfall']
    
    # Prepare input features
    feature_ls = [N,P,K,temp,hum,ph,rain]
    single_pr = np.array(feature_ls).reshape(1, -1)
    # Transform features using preprocessing pipeline
    scaled_features = mx.transform(single_pr)
    sc_features = sc.transform(scaled_features)
    final_features = pre_pro.transform(sc_features)
    
    # Make prediction using the model
    prediction = model.predict(final_features)
    
    # Dictionary mapping crop numbers to crop names
    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange", 8: "Apple",
                 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate",
                 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
                 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    # Generate result message
    if prediction[0] in crop_dict:
        data = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated.".format(data)
    else:
        result = "Sorry, unable to recommend a proper crop for this environment."

    return render_template('index.html', result=result)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
