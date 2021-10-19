import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from datetime import date
import scrape

app = Flask(__name__)
model = pickle.load(open('model1.pkl','rb'))
rice = pickle.load(open('rice.pkl','rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	ricepic=False
	cottonpic=False
	maizepic=False
	msp=0;
	datenow=date.today().strftime('%d/%m/%Y')
	int_features = [float(x) for x in request.form.values()]
	#print(int_features)
	final_features = [np.array(int_features)]
	prediction = model.predict(final_features)
	if(prediction[0]=='rice'):
		#today=datetime.today()
		#d1 = int(today.strftime("%d/%m/%Y"))
		#d1=int(today)
		#print(d1)
		msp=scrape.paddy
		ricepic=True
		#pred[5]=round(ricepred[0],2)
		#print(ricepred)
	elif(prediction[0]=='cotton'):
		msp=scrape.cotton
		cottonpic=True
		ricepred = rice.predict(np.array(120).reshape(1, -1))
		#print(msp)
	elif(prediction[0]=='maize'):
		msp=scrape.maize
		maizepic=True
		ricepred = rice.predict(np.array(120).reshape(1, -1))
		#print(msp)
	elif(prediction[0]=='blackgram'):
		msp=scrape.urad
		#print(msp)
	elif(prediction[0]=='jute'):
		msp=scrape.jute
	return render_template('index.html',crop=prediction[0],msp=msp,date=datenow,rice=ricepic,cotton=cottonpic,maize=maizepic);

@app.route('/predict_api',methods=['POST'])
def predict_api():
	data = request.get_json(force=True)
	prediction = model.predict([np.array(list(data.values()))])
	output = prediction[0]
	return jsonify(output)

if __name__ == "__main__":
	app.run(debug=True)