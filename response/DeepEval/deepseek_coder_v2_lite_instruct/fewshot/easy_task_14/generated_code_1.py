import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(input_layer)

    # Fully connected layers to generate weights
    fc1 = Dense(units=128, activation='relu')(gap)
    weights = Dense(units=32 * 32 * 3, activation='sigmoid')(fc1)

    # Reshape weights to align with input shape
    reshaped_weights = Reshape(target_shape=(32, 32, 3))(weights)

    # Element-wise multiplication with input feature map
    multiplied = Multiply()([input_layer, reshaped_weights])

    # Flatten the result
    flattened = Flatten()(multiplied)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model