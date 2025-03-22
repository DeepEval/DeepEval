from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, Reshape, Multiply, Concatenate
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D

def dl_model():

    # Input layer
    input_img = Input(shape=(32, 32, 3))

    # Convolutional layer
    x = Conv2D(filters=16, kernel_size=3, padding='same')(input_img)

    # Batch normalization and ReLU activation
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Max pooling
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Fully connected layers
    x = Dense(units=8, activation='relu')(x)
    x = Dense(units=16, activation='relu')(x)

    # Reshape and multiply with initial features
    x = Reshape((1, 1, 16))(x)
    weighted_features = Multiply()([x, x])

    # Concatenate weighted features with input layer
    combined_features = Concatenate()([input_img, weighted_features])

    # 1x1 convolution and average pooling
    x = Conv2D(filters=16, kernel_size=1, padding='same')(combined_features)
    x = AveragePooling2D(pool_size=2, strides=2)(x)

    # Fully connected output layer
    output = Dense(units=10, activation='softmax')(x)

    # Model creation
    model = Model(inputs=input_img, outputs=output)

    return model