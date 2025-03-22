import keras
from keras.layers import Input, Lambda, Conv2D, Concatenate, Flatten, Dense, SeparableConv2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input image into three channel groups
    channel_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Perform feature extraction for each group
    path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_groups[0])
    path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(channel_groups[1])
    path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(channel_groups[2])

    # Concatenate the outputs from the three groups
    concat = Concatenate(axis=-1)([path1, path2, path3])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concat)

    # Pass through three fully connected layers
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    dense2 = Dense(units=32, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model