import keras
from keras.layers import Input, Lambda, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split channels into three groups
    split_channels = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)

    # Apply different kernels to each group
    group1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_channels[0])
    group2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split_channels[1])
    group3 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(split_channels[2])

    # Concatenate outputs from each group
    merged_features = Concatenate()( [group1, group2, group3] )

    # Flatten and add fully connected layers
    flatten_layer = Flatten()(merged_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model