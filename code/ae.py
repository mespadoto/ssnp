import os

import numpy as np
import tensorflow.keras.backend as K
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


class AutoencoderProjection(BaseEstimator, TransformerMixin):
    def __init__(self, epochs=100, verbose=0):
        self.epochs = epochs
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.verbose = verbose
        self.is_fitted = False

        K.clear_session()


    def fit(self, X):
        ae_input = Input(shape=(X.shape[1],))
        encoded = Dense(512, activation='relu')(ae_input)
        encoded = Dense(128, activation='relu')(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(2, activation='linear')(encoded)
        decoded = Dense(32, activation='relu', name='enc1')(encoded)
        decoded = Dense(128, activation='relu', name='enc2')(decoded)
        decoded = Dense(512, activation='relu', name='enc3')(decoded)
        decoded = Dense(X.shape[1], activation='sigmoid', name='decoder_output')(decoded)

        self.encoder = Model(inputs=ae_input, outputs=encoded)

        self.autoencoder = Model(ae_input, decoded)
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        self.autoencoder.fit(   X, 
                                X,
                                epochs=self.epochs,
                                batch_size=32,
                                shuffle=True,
                                verbose=self.verbose)

        encoded_input = Input(shape=(2,))
        l = self.autoencoder.get_layer('enc1')(encoded_input)
        l = self.autoencoder.get_layer('enc2')(l)
        l = self.autoencoder.get_layer('enc3')(l)
        decoder_layer = self.autoencoder.get_layer('decoder_output')(l)

        self.decoder = Model(encoded_input, decoder_layer)

        self.is_fitted = True

    def transform(self, X):
        if self._is_fit():
            return self.encoder.predict(X)
           
    def inverse_transform(self, X_2d):
        if self._is_fit():
            return self.decoder.predict(X_2d)

    def _is_fit(self):
        if self.is_fitted:
            return True
        else:
            raise Exception('Model not trained. Call fit() before calling transform()')
