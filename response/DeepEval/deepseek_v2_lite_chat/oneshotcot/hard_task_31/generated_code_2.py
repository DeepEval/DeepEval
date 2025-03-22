import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.models import Model

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    main_path_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(main_path_output)
    main_path_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_output)
    main_path_output = BatchNormalization()(main_path_output)
    main_path_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_path_output)

    branch_path_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Concatenate outputs from both paths
    concat = Concatenate()([main_path_output, branch_path_output])

    # Block 2
    def block(input_tensor):
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Dropout(rate=0.5)(x)

        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Dropout(rate=0.5)(x)

        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Dropout(rate=0.5)(x)

        x = Flatten()(x)

        return x

    conv1 = block(concat)
    conv2 = block(conv1)
    conv3 = block(conv2)

    # Concatenate outputs from the three groups
    final_output = Concatenate()([conv1, conv2, conv3])

    # Fully connected layers
    dense1 = Dense(units=512, activation='relu')(final_output)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model