import keras
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input channels into three groups
    split_tensor = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)
    
    # Apply 1x1 convolutions to each group
    group1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    group2 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    group3 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])

    # Average pooling for downsampling
    pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(group1)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(group2)
    pool3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(group3)

    # Concatenate the pooled feature maps
    concat_layer = Concatenate(axis=2)([pool1, pool2, pool3])

    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model