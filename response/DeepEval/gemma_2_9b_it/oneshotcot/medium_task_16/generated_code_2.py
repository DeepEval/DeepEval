import keras
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_tensor = Input(shape=(32, 32, 3)) 

    # Split the input channels
    split_tensor = Lambda(lambda x: tf.split(x, 3, axis=2))(input_tensor)
    
    # Process each group independently
    group1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    group2 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    group3 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])

    # Downsample each group
    group1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(group1)
    group2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(group2)
    group3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(group3)

    # Concatenate the results
    merged_features = Concatenate(axis=2)([group1, group2, group3])

    # Flatten and classify
    flatten_layer = Flatten()(merged_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_tensor, outputs=output_layer)

    return model