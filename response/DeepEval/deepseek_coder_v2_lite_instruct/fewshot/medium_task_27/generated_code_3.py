import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply, concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First branch with 3x3 convolutional layer
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv_3x3)

    # Second branch with 5x5 convolutional layer
    conv_5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    conv_5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(conv_5x5)

    # Adding the outputs of both branches
    added = Add()([conv_3x3, conv_5x5])

    # Global average pooling
    gap = GlobalAveragePooling2D()(added)

    # Fully connected layers
    fc1 = Dense(units=64, activation='relu')(gap)
    fc2 = Dense(units=10, activation='softmax')(fc1)  # 10 classes in CIFAR-10

    # Attention mechanism
    attention_weights = Dense(units=2, activation='softmax')(fc1)  # Two weights for two branches
    attention_weights_expanded = keras.backend.expand_dims(attention_weights, axis=1)
    attention_weights_expanded = keras.backend.expand_dims(attention_weights_expanded, axis=1)

    # Applying attention weights to branch outputs
    weighted_3x3 = Multiply()([conv_3x3, attention_weights[:, 0:1, 0:1, :]])
    weighted_5x5 = Multiply()([conv_5x5, attention_weights[:, 1:2, 0:1, :]])

    # Combining the weighted outputs
    combined = Add()([weighted_3x3, weighted_5x5])

    # Final output
    final_output = Dense(units=10, activation='softmax')(combined)

    model = Model(inputs=input_layer, outputs=final_output)

    return model