from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Dense, Activation, Dropout, Add, Permute, Conv2DTranspose
from keras.layers import Layer
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
import keras.backend as K
import numpy as np
from keras.datasets import cifar10

def dl_model():
    input_shape = (32, 32, 3)  # Input image size (example for CIFAR-10)
    num_classes = 10  # Number of classes in CIFAR-10

    # Input layer
    input_layer = Input(shape=input_shape)

    # Block 1
    x = Conv2D(16, (3, 3), padding='same')(input_layer)  # Adjust number of channels to match input
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Path 1: Global average pooling
    avg_pool = GlobalAveragePooling2D()(x)
    path1 = Dense(128, activation='relu')(avg_pool)
    path1 = Dense(64, activation='relu')(path1)

    # Path 2: Global max pooling
    max_pool = GlobalMaxPooling2D()(x)
    path2 = Dense(128, activation='relu')(max_pool)
    path2 = Dense(64, activation='relu')(path2)

    # Add and pass through activation for channel attention
    combined = Add()([path1, path2])
    attention = Activation('sigmoid')(Dense(x.shape[1], activation='sigmoid')(combined))

    # Element-wise multiplication with original features
    x = Permute((2, 1))(x)
    x = K.dot(x, K.permute_dimensions(attention, (0, 2, 1)))
    x = Activation('relu')(x)

    # Block 2: Average and max pooling
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)
    concatenated = Concatenate()([avg_pool, max_pool])
    x = Conv2D(64, (1, 1), padding='same')(concatenated)
    x = Activation('sigmoid')(x)

    # Element-wise multiplication with Path1 outputs
    x = Permute((2, 1))(x)
    x = K.dot(x, K.permute_dimensions(attention, (0, 2, 1)))

    # Additional branch for aligning output channels
    additional = Conv2D(16, (1, 1), padding='same')(x)
    additional = Activation('relu')(additional)
    additional = Conv2D(3, (1, 1), padding='same')(additional)
    additional = Activation('sigmoid')(additional)

    # Final output
    output = Add()([x, additional])
    output = Dense(num_classes, activation='softmax')(output)

    # Model construction
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Instantiate and return the model
model = dl_model()
model.summary()