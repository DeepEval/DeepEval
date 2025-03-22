import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Global Average Pooling Layer
    gap_layer = GlobalAveragePooling2D()(input_layer)

    # Fully Connected Layer to generate weights
    fc_layer = Dense(units=32, activation='relu')(gap_layer)  # 32 units as an example, can be adjusted
    weights = Dense(units=32, activation='linear')(fc_layer)  # Linear activation to keep the size

    # Reshape weights to align with input shape
    reshape_weights = keras.layers.Reshape((1, 1, 32))(weights)

    # Multiply the weights with the input feature map
    multiplied_map = Multiply()([input_layer, reshape_weights])

    # Flatten the result
    flattened_layer = Flatten()(multiplied_map)

    # Fully Connected Layer for final classification
    output_layer = Dense(units=10, activation='softmax')(flattened_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model