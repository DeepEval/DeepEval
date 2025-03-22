import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Concatenate, Conv2D, Reshape, AveragePooling2D
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Path 1: Max pooling 1x1
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    path1 = Flatten()(path1)
    path1 = Dropout(0.5)(path1)

    # Path 2: Max pooling 2x2
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    path2 = Flatten()(path2)
    path2 = Dropout(0.5)(path2)

    # Path 3: Max pooling 4x4
    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    path3 = Flatten()(path3)
    path3 = Dropout(0.5)(path3)

    # Concatenate the outputs from Block 1
    block1_output = Concatenate()([path1, path2, path3])
    
    # Fully connected layer
    fc_layer = Dense(units=128, activation='relu')(block1_output)

    # Reshape for Block 2 (reshape to (batch_size, height, width, channels))
    reshaped_output = Reshape((4, 4, 8))(fc_layer)  # Example shape; adjust according to your needs

    # Block 2
    # Path 1: 1x1 Convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)

    # Path 2: 1x1 -> 1x7 -> 7x1 Convolutions
    path2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    path2 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 -> (7x1 <-> 1x7) Convolutions
    path3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    path3 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(path3)

    # Path 4: Average pooling followed by 1x1 Convolution
    path4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(reshaped_output)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path4)

    # Concatenate the outputs from Block 2
    block2_output = Concatenate()([path1, path2, path3, path4])

    # Flatten and output layer
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=dense2)

    return model