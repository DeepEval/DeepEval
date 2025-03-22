import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    max_pooling = MaxPooling2D((2, 2))(conv2)

    # Branch path
    conv3 = Conv2D(64, (5, 5), activation='relu')(input_layer)

    # Combine features
    merged = Concatenate()([max_pooling, conv3])

    # Flatten and map to probability distribution
    flattened = Flatten()(merged)
    dense1 = Dense(128, activation='relu')(flattened)
    output_layer = Dense(10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model