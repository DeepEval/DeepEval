import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block with main and branch paths
    # Main Path
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='relu')(main_conv1)

    # Branch Path
    branch_path = input_layer

    # Combine both paths using addition
    block1_output = Add()([main_conv2, branch_path])

    # Second block with three max pooling layers
    # Max pooling layers with different sizes
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(block1_output)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block1_output)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(block1_output)

    # Flatten the outputs
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)

    # Concatenate the flattened outputs
    concatenated_output = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model