import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
from keras.models import load_model
import numpy as np
class ClassficationModel():
    def __init__(self):
        self.model = None

    def __build_model(self, num_features):
        model = tf.keras.Sequential([
            layers.GaussianNoise(0.15),
            layers.Dense(128, input_dim=num_features, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.6),
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.6),
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.6),
            layers.Dense(4)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.01),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )
        return model
    
    def model_fit(self, X_train, y_train):
        self.model = self.__build_model(X_train.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

        def scheduler(epoch, lr):
            return lr * 0.98

        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

        callbacks = [early_stopping, model_checkpoint, learning_rate_scheduler]

        self.model.fit(
            X_train
            , y_train
            , batch_size=32
            , epochs=500
            , validation_split=0.3
            , callbacks=callbacks
        )
    def evaluate(self, X_test, y_test):
        self.model = load_model('best_model.h5')
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test set accuracy: {accuracy}")
        return accuracy
    
    def predict(self, X_test):
        self.model = load_model('best_model.h5')
        predictions = self.model.predict(X_test)
        predictions = np.argmax(predictions, axis=1)
        return predictions
    