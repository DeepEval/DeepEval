import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First branch: 3x3 convolutional layer
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3x3)

    # Second branch: 5x5 convolutional layer
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv5x5)

    # Add the outputs of the two branches
    added = Add()([conv3x3, conv5x5])

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(added)

    # Fully connected layers
    fc1 = Dense(units=64, activation='relu')(gap)
    fc2 = Dense(units=10, activation='softmax')(fc1)

    # Attention mechanism
    attention_weights = Dense(units=2, activation='softmax')(fc1)
    attention_weights_expanded = keras.layers.expand_dims(attention_weights, axis=1)
    attention_weights_expanded = keras.layers.expand_dims(attention_weights_expanded, axis=1)

    # Weight the outputs of the branches
    weighted_output1 = Multiply()([conv3x3, attention_weights_expanded[:, :, :, 0]])
    weighted_output2 = Multiply()([conv5x5, attention_weights_expanded[:, :, :, 1]])

    # Add the weighted outputs
    final_output = Add()([weighted_output1, weighted_output2])

    # Final model
    model = Model(inputs=input_layer, outputs=final_output)

    return model