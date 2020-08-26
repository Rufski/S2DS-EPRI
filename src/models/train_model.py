import numpy as np
import pandas as pd
import tensorflow as tf

def train_model(model,
                X_train,
                y_train,
                X_valid,
                y_valid,
                run_index,
                model_name,
                log_dir_path,
                checkpoint_dir_path,
                nr_epochs,
                save_freq=100,        
                patience=300):
    """
    Train a keras model.

        Args:
            model (keras model): model to train
            X_train (np.array): an array containg all of the training sets
            y_train (np.array): an array containg corresponding response variables 
            X_valid (np.array): an array containg all of the validation sets
            y_valid (np.array): an array containg corresponding response variables 
            run_index (int): index to separate different runs
            model_name (str): name of the current model
            log_dir_path (str): path to the dirctory where the logs will be stored
            checkpoint_dir_path (str): path to the directory where the
                checkpoints will be stored, the directory must exist
            nr_epochs (int): number of epochs to train for
            save_freq (int, optional): number of batches after which the model is saved
            patience (int, optional): patience value for early stopping

        Return:
            str: the path to the checkpoint
    """
    # Set the relevant callbacks    
    log_path = log_dir_path + "/" + model_name + f"_run{run_index:03d}"  
    checkpoint_path = checkpoint_dir_path + "/" + model_name + f"_run{run_index:03d}.h5"

    if nr_epochs < save_freq:
        save_freq = nr_epochs

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=patience),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_freq=save_freq),
        tf.keras.callbacks.TensorBoard(log_dir=log_path),
    ]
    
    
    # Train the model
    model.fit(X_train, 
              y_train,
              epochs=nr_epochs,
              validation_data=(X_valid, y_valid),
              callbacks=my_callbacks)

    return checkpoint_path
