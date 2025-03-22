import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Split the input along the channel dimension
    def split_channels(tensor):
      split_tensors = tf.split(tensor, num_or_size_splits=3, axis=2) 
      return split_tensors

    split_tensors = Lambda(split_channels)(input_layer)

    # Apply different kernels to each group
    group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensors[0])
    group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensors[1])
    group3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensors[2])

    # Concatenate the outputs from the three groups
    merged_features = Concatenate()( [group1, group2, group3])

    # Flatten the features
    flatten_layer = Flatten()(merged_features)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model