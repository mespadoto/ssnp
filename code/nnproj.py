from sklearn import decomposition, preprocessing
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, optimizers

class NNProj():
    def __init__(self, init=decomposition.PCA(n_components=2), size='medium', style='bottleneck', loss='mean_absolute_error', epochs=1000, opt='adam', l1=0.0, l2=0.0, dropout=True):
        self.nnsettings = dict()

        self.nnsettings['small'] = dict()
        self.nnsettings['medium'] = dict()
        self.nnsettings['large'] = dict()
        self.nnsettings['std'] = dict()

        self.nnsettings['std']['wide'] = [256,512,256]

        self.nnsettings['small']['straight'] = [120,120,120]
        self.nnsettings['small']['wide'] = [90,180,90]
        self.nnsettings['small']['bottleneck'] = [150,60,150]

        self.nnsettings['medium']['straight'] = [240,240,240]
        self.nnsettings['medium']['wide'] = [180,360,180]
        self.nnsettings['medium']['bottleneck'] = [300,120,300]

        self.nnsettings['large']['straight'] = [480,480,480]
        self.nnsettings['large']['wide'] = [360,720,360]
        self.nnsettings['large']['bottleneck'] = [600,240,600]
        self.layers = self.nnsettings[size][style]

        self.stop = EarlyStopping(verbose=0, min_delta=0.00001, mode='min', patience=10, restore_best_weights=True)
        self.callbacks = [self.stop]

        self.init = init
        self.dropout = dropout
        self.opt = opt
        self.epochs = epochs
        self.loss = loss
        self.l1 = l1
        self.l2 = l2

        self.is_fitted = False
        K.clear_session()

    def fit(self, X):
        self.model = Sequential()
        self.model.add(Dense(self.layers[0], activation='relu',
                    kernel_initializer='he_uniform',
                    bias_initializer=Constant(0.0001),
                    input_shape=(X.shape[1],)))
        self.model.add(Dense(self.layers[1], activation='relu',
                    kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
                    kernel_initializer='he_uniform',
                    bias_initializer=Constant(0.0001)))
        self.model.add(Dense(self.layers[2], activation='relu',
                    kernel_initializer='he_uniform',
                    bias_initializer=Constant(0.0001)))
        if self.dropout:
            self.model.add(Dropout(0.5))

        self.model.add(Dense(2, activation='sigmoid',
                    kernel_initializer='he_uniform',
                    bias_initializer=Constant(0.0001)))
        self.model.compile(loss=self.loss, optimizer=self.opt)

        X_2d = self.init.fit_transform(X)
        scaler = preprocessing.MinMaxScaler()
        X_2d = scaler.fit_transform(X_2d)

        self.model.fit(X, X_2d, batch_size=32, epochs=self.epochs, verbose=0, validation_split=0.05, callbacks=self.callbacks)
        self.is_fitted = True

    def _is_fit(self):
        if self.is_fitted:
            return True
        else:
            raise Exception('Model not trained. Call fit() before calling transform()')

    def transform(self, X):
        if self._is_fit():
            return self.model.predict(X)
