import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, Reshape, Permute, DepthwiseConv2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Branch path
    branch_path = AveragePooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(input_layer)

    # Main path
    x = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    x = [Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(i) for i in x]
    x = Concatenate()(x)  

    # Block 2
    shape = Lambda(lambda x: tf.shape(x))(x)
    x = Reshape(target_shape=(shape[1], shape[2], 3, 32))(x) 
    x = Permute(dims=[0, 1, 3, 2])(x)
    x = Reshape(target_shape=(shape[1], shape[2], 96))(x)

    # Block 3
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    # Combine paths and classify
    x = Concatenate()([x, branch_path])
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model