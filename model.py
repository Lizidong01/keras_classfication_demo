import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

class ClassificationModel:
    def __init__(self, num_classes, optimizer=tf.keras.optimizers.Adam, learning_rate=0.01,
                 hidden_units=(128, 32, 32), dropout_rates=(0.6, 0.6, 0.6),
                 l2_regularizer=0.01):
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.dropout_rates = dropout_rates
        self.l2_regularizer = l2_regularizer

        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential()

        model.add(layers.GaussianNoise(0.15))

        for units, dropout_rate in zip(self.hidden_units, self.dropout_rates):
            model.add(layers.Dense(units, activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularizer)))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(self.num_classes))

        model.compile(
            optimizer=self.optimizer(learning_rate=self.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        return model

    def fit(self, X_train, y_train, batch_size=32, epochs=500, validation_split=0.3, callbacks=None):
        if callbacks is None:
            callbacks = [EarlyStopping(monitor='val_loss', patience=20),
                         ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)]

        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                       validation_split=validation_split, callbacks=callbacks)

    def evaluate(self, X_test, y_test):
        self.model = models.load_model('best_model.h5')
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test set accuracy: {accuracy}")
        return accuracy

    def predict(self, X_test):
        self.model = models.load_model('best_model.h5')
        predictions = self.model.predict(X_test)
        predictions = np.argmax(predictions, axis=1)
        return predictions

    def save_model(self, model_name):
        # self.model.save(model_name)
        self.model.save(model_name + ".h5")
