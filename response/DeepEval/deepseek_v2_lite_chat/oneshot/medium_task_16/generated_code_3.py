import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split1, split2, split3 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # 1x1 convolution for the first group
    conv1 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(split1)
    # 1x1 convolution for the second group
    conv2 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(split2)
    # 1x1 convolution for the third group
    conv3 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(split3)

    # Average pooling for each group
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

    # Concatenate along the channel dimension
    concat = Concatenate(axis=-1)([pool1, pool2, pool3])

    # Flatten and pass through fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate and print the model
model = dl_model()
model.summary()