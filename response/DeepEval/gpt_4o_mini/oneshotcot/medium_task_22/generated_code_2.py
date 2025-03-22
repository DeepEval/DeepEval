import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, BatchNormalization

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: First branch - 3x3 convolutions
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Step 3: Second branch - 1x1 convolutions followed by two 3x3 convolutions
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

    # Step 4: Third branch - max pooling
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_layer)

    # Step 5: Concatenate the outputs from all branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Step 6: Batch normalization
    batch_norm = BatchNormalization()(concatenated)

    # Step 7: Flatten the result
    flatten_layer = Flatten()(batch_norm)

    # Step 8: Add first dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Step 9: Add second dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Step 10: Output layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()