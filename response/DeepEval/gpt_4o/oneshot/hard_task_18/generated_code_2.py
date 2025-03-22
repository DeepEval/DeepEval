import keras
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Add, Multiply, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2), padding='same')(conv2)

    # Adding input to the output of the main path
    added = Add()([avg_pool, input_layer])

    # Second Block
    global_avg_pool = GlobalAveragePooling2D()(added)
    dense1 = Dense(units=64, activation='relu')(global_avg_pool)
    dense2 = Dense(units=added.shape[-1], activation='sigmoid')(dense1)  # Same number of channels as added
    channel_weights = Multiply()([added, dense2])

    # Flatten and Fully Connected layer for classification
    flatten_layer = Flatten()(channel_weights)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate the model
model = dl_model()