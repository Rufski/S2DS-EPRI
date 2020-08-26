import tensorflow as tf

def load_and_predict_model(path_checkpoint, X):
    model = tf.keras.models.load_model(path_checkpoint)
    return model.predict(X)
