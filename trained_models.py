import numpy as np
import xgboost as xgb
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

class KerasModel:
    def __init__(self, language: str, model_dir_template: str = r'models\{language}\keras_ANN.keras'):
        self.model_dir = model_dir_template.format(language=language)
        self.model = keras.models.load_model(self.model_dir)

    def detect(self, input: list):
        x = np.array(input).reshape(1, -1)
        label = np.argmax(self.model.predict(x), axis=-1)[0]
        return chr(int(label) + 65)
    
class XGBoostModel:
    def __init__(self, language: str, model_dir_template: str = r'models\{language}\xgboost_model.json'):
        self.model_dir = model_dir_template.format(language=language)
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.model_dir)
    
    def detect(self, input: list):
        x = np.array(input).reshape(1, -1)
        label = self.model.predict(x)
        return chr(int(label) + 65)
