import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Global Average Pooling to capture global information
    global_avg_pool = GlobalAveragePooling2D()(input_layer)

    # Fully connected layers to generate channel weights
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=3, activation='sigmoid')(dense1)  # Assuming 3 channels for RGB

    # Reshape to align with input shape
    channel_weights = Reshape((1, 1, 3))(dense2)

    # Multiply channel weights with the input feature map
    weighted_input = Multiply()([input_layer, channel_weights])

    # Flatten the weighted input and pass through a fully connected layer for classification
    flatten = Flatten()(weighted_input)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model