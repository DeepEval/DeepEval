from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from keras.models import Model
import keras.backend as K

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    groups = Lambda(lambda x: K.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply 1x1 convolutions to each group independently
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(groups[0])
    conv2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(groups[1])
    conv3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(groups[2])

    # Apply average pooling to each group
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    # Concatenate the pooled feature maps
    pool_concat = Concatenate(axis=-1)([pool1, pool2, pool3])

    # Flatten the concatenated feature maps
    flatten = Flatten()(pool_concat)

    # Add a dense layer with 64 units and activation function 'relu'
    dense1 = Dense(units=64, activation='relu')(flatten)

    # Add a dense layer with 32 units and activation function 'relu'
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Add a dense layer with 10 units and activation function 'softmax' for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model by passing the output layer to the Model class
    model = Model(inputs=input_layer, outputs=output_layer)

    return model