import keras
from keras.layers import Input, AveragePooling2D, Conv2D, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Average pooling layer with 5x5 window and 3x3 stride
    avg_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(input_layer)

    # 1x1 convolutional layer
    conv1x1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(avg_pool)

    # Flatten the feature maps
    flatten = Flatten()(conv1x1)

    # Two fully connected layers with dropout
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(Dropout(rate=0.2)(dense1))

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model