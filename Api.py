import flask
from flask import request
app = flask.Flask(__name__)
app.config["DEBUG"] = True

from flask_cors import CORS
CORS(app)


# main index page route
@app.route('/')
def home():
    return '<h1>API is working.. </h1>'



@app.route('/predict', methods=['GET'])
def predict():
    import joblib
    model1 = joblib.load('creditscore_predict_model.pkl')
    credictscore_predict = model1.predict([[request.args['partner_name'], 
                                            request.args['days_late'], 
                                            request.args['call_contact_status'], 
                                            request.args['type'], 
                                            request.args['status'], 
                                            request.args['balance_remaining'], 
                                            request.args['amount_already_paid']
                                           ]])  
    
    return str(credictscore_predict[0])


if __name__ == "__main__":
    app.run(debug=True)
