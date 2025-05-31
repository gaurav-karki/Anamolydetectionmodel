import pickle

class AnamolyDetection:
	def __init__(self):
		#load the model during initialization
		with open('../notebooks/logistic_regression_anamolydetectionl.pkl','rb') as f:
			self.model =pickle.load(f)


	def predict_anomaly(self, X):
		# use the loaded model to make a prediction
		return self.model.predict(X)
		
