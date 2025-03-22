import keras
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Flatten, Add, Reshape, Multiply
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: First Block - Two 3x3 Convolutions followed by Average Pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    avg_pooling = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)

    # Step 3: Add input to output of the first block via addition
    added_output = Add()([input_layer, avg_pooling])

    # Step 4: Second Block - Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(added_output)

    # Step 5: Two Fully Connected Layers
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Step 6: Reshape weights for multiplication
    reshaped_weights = Reshape((1, 1, 32))(dense2)

    # Step 7: Multiply the reshaped weights by the input
    multiplied_output = Multiply()([added_output, reshaped_weights])

    # Step 8: Flatten the output
    flatten_layer = Flatten()(multiplied_output)

    # Step 9: Fully Connected Layer for Classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model