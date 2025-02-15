import numpy as np
import xgboost as xgb
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

class KerasModel:
    def __init__(self, model_dir: str = r'models\keras_ANN.keras'):
        self.model = keras.models.load_model(model_dir)
        self.scaler = StandardScaler()
    
    def detect(self, input: list):
        x = np.array(input).reshape(1, -1)
        # x_scaled = self.scaler.transform(x)
        label = np.argmax(self.model.predict(x), axis=-1)[0]
        return chr(int(label) + 65)
    
class XGBoostModel:
    def __init__(self, model_dir: str = r'models\xgboost_model.json'):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_dir)
        self.scaler = StandardScaler()
    
    def detect(self, input: list):
        x = np.array(input).reshape(1, -1)
        # x_scaled = self.scaler.transform(x)
        # print(x_scaled)
        label = self.model.predict(x)
        return chr(int(label) + 65)
