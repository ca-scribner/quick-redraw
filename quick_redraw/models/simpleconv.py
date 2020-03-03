from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from ray import tune

SEARCH_PARAMS = {
    'conv_filters': tune.grid_search([32]),
    'conv_kernel_size': tune.grid_search([3]),
    'conv_stride': tune.grid_search([1]),
    'pool_size': tune.grid_search([2]),
    'dense_layer_size': tune.grid_search([64, 128]),
    'dropout': tune.grid_search([0.0, 0.2]),
    'learning_rate': tune.grid_search([0.0001])
    # 'epochs': tune.grid_search([1, 5, 10, 20])  # Training epochs.  Bother varying?  What would this affect?
    # 'dense_layer_size': tune.grid_search([64, 128]),
    # 'dropout': tune.grid_search([0.0, 0.2, 0.5]),
    # 'learning_rate': tune.grid_search([0.0001, 0.001, 0.01])
}


class SimpleConv(Model):
    def __init__(self, n_output, conv_filters=28, conv_kernel_size=None, conv_stride=1, pool_size=None,
                 dense_layer_size=128, dropout=0.0):
        if not conv_kernel_size:
            conv_kernel_size = (2, 2)

        super().__init__()
        self.layers_ = []
        self.layers_.append(Conv2D(filters=conv_filters,
                                   kernel_size=conv_kernel_size,
                                   strides=conv_stride))
        if pool_size:
            self.layers_.append(MaxPooling2D(pool_size=pool_size))
        self.layers_.append(Flatten())
        self.layers_.append(Dense(dense_layer_size, activation="relu"))
        self.layers_.append(Dropout(dropout))

        # Output layer
        self.layers_.append(Dense(n_output, activation='softmax'))

    def call(self, x):
        # FUTURE: super signature has training=None, mask=None.  Fix
        for layer in self.layers_:
            x = layer(x)
        return x
