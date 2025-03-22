import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Concatenate, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 input shape

    # Step 1: Split the input into three groups along the channel dimension
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Step 2: Apply 1x1 convolutions to each group
    conv1 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
    conv2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
    conv3 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
    
    # Step 3: Downsample each group using average pooling
    pooled1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    pooled2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    pooled3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    # Step 4: Concatenate the pooled outputs along the channel dimension
    concatenated = Concatenate()([pooled1, pooled2, pooled3])

    # Step 5: Flatten the concatenated feature maps
    flatten_layer = Flatten()(concatenated)

    # Step 6: Pass through fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model