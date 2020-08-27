import tensorflow as tf

def load_and_predict_model(path_checkpoint, X, custom_func=None):
    if custom_func is not None:
        tf.keras.utils.get_custom_objects().update(custom_func)
    model = tf.keras.models.load_model(path_checkpoint)
    return model.predict(X)
