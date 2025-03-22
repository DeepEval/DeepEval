import keras
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 3)) 

    # Split the input into three groups along the channel dimension
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(input_layer)

    # Apply 1x1 convolutions to each group
    group1 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split_tensor[0])
    group2 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split_tensor[1])
    group3 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split_tensor[2])

    # Apply average pooling to each group
    pool1 = AveragePooling2D(pool_size=(2, 2))(group1)
    pool2 = AveragePooling2D(pool_size=(2, 2))(group2)
    pool3 = AveragePooling2D(pool_size=(2, 2))(group3)

    # Concatenate the three groups along the channel dimension
    concatenated = Concatenate(axis=2)([pool1, pool2, pool3])

    # Flatten the concatenated feature maps
    flatten_layer = Flatten()(concatenated)

    # Add two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model