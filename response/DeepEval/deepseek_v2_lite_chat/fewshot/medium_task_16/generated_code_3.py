import keras
from keras.layers import Input, Lambda, Conv2D, MaxPool2D, Flatten, Dense

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups along the channel dimension
    split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply 1x1 convolutions to each group and downsample with average pooling
    conv1 = [Conv2D(filters=int(x/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(i) for i in split1]
    pool1 = [MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(i) for i in conv1]

    # Concatenate feature maps along the channel dimension
    concat = keras.layers.Concatenate()(pool1)

    # Flatten concatenated feature maps
    flatten = Flatten()(concat)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model