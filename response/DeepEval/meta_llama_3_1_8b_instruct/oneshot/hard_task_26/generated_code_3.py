import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add
from keras.layers import UpSampling2D, Reshape

def dl_model():
    
    # Define the input shape for the model
    input_shape = (32, 32, 3)
    input_layer = Input(shape=input_shape)

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Branch 2: Downsample, Convolution, Upsample
    conv2_down = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_down)
    upsample2 = UpSampling2D(size=(2, 2))(conv2)

    # Branch 3: Downsample, Convolution, Upsample
    conv3_down = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_down)
    upsample3 = UpSampling2D(size=(2, 2))(conv3)

    # Concatenate the outputs from all branches
    concat = Concatenate()([branch1, upsample2, upsample3])
    conv4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)

    # Branch path
    conv5 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the outputs from the main path and the branch path
    add = Add()([conv4, conv5])

    # Flatten the output
    flatten_layer = Flatten()(add)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model