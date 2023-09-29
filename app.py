# from pyngrok import ngrok
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer

app = Flask(__name__)
CORS(app)

root_top = "/home/ubuntu/duy/Telemec_AI_server/"
# Disease Model
rf_ds_model = joblib.load(root_top + "Disease Model" + '/random_forest_model.pkl')
nb_ds_model = joblib.load(root_top + "Disease Model" + '/nb_model.pkl')
svm_ds = joblib.load(root_top + "Disease Model" + '/svm.pkl')
# Advisor Model
rf_model = joblib.load(root_top + "Advisor Model" + '/random_forest_model.pkl')
dt_model = joblib.load(root_top + 'Advisor Model' + '/decision_tree_model.pkl')
nb_model = joblib.load(root_top + 'Advisor Model' + '/nb_model.pkl')
svm = joblib.load(root_top + 'Advisor Model' + '/svm.pkl')

@app.route("/")
def home():
    return "Hello, worlddddddddddddd!"

@app.route("/advisor", methods=["POST"])
def index_predict():
    try:
      data = request.json
      new_data = [[
        float(data['SBP']),
        float(data['DBP']),
        float(data['heart_rate']),
        float(data['Glucose']),
        float(data['SpO2']),
        float(data['TemperatureInF'])
      ]]

      rf_result = rf_model.predict(new_data)[0]
      dt_result = dt_model.predict(new_data)[0]
      nb_result = nb_model.predict(new_data)[0]
      svm_result = svm.predict(new_data)[0]
      return jsonify({'rf_result': int(rf_result),'dt_result':int(dt_result),'nb_result':int(nb_result),'svm_result':int(svm_result)})
    except Exception as e:
      app.logger.error(f"An error occurred during prediction: {str(e)}")
      return jsonify({'error': 'An error occurred during prediction.'}), 500

@app.route("/disease", methods=["POST"])
def index_disease():
    try:
      data = request.json
      new_data = [[value for key, value in data.items()]]
      rf_result = rf_ds_model.predict(new_data)[0]
      nb_result = nb_ds_model.predict(new_data)[0]
      svm_result = svm_ds.predict(new_data)[0]

      return jsonify({'rf_result': rf_result,'nb_result':nb_result,'svm_result':svm_result})
    except Exception as e:
      app.logger.error(f"An error occurred during prediction: {str(e)}")
      return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == "__main__":
    app.run(host = "0.0.0.0")