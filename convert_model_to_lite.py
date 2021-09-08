from tensorflow import lite
from tensorflow.keras import models


model = models.load_model('final_model.h5')
converter = lite.TFLiteConverter.from_keras_model(model)
tflit_model = converter.convert()
open('tflite_final_model.tflite', 'wb').write(tflit_model)