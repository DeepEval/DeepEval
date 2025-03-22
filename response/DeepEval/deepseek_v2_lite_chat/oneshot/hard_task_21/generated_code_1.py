import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    split_layer = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    split_layer = [Lambda(lambda y: keras.backend.mean(y, axis=2))(x) for x in split_layer]
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])

    # Concatenate the outputs from the main path
    main_output = Concatenate()([conv1, conv2, conv3])

    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add the branch path to the main path
    model_output = Concatenate()([main_output, branch_conv])

    # Batch normalization and flattening
    batch_norm_output = BatchNormalization()(model_output)
    flatten_layer = Flatten()(batch_norm_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()