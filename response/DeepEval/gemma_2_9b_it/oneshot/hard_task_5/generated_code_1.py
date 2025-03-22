import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Permute
from tensorflow.keras.layers import tf

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))

        # Block 1
        x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
        x = [Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(y) for y in x]
        x = Concatenate(axis=3)(x)

        # Block 2
        shape_tensor = Lambda(lambda x: tf.shape(x))(x)
        x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(x)
        x = [Reshape(target_shape=(shape_tensor[1], shape_tensor[2], 3, x_shape[-1] // 3))(y) for x, x_shape in zip(x, shape_tensor)]
        x = [Permute((2, 3, 1, 4))(y) for y in x]
        x = [Reshape(target_shape=(shape_tensor[1], shape_tensor[2], shape_tensor[3]))(y) for y in x]
        x = Concatenate(axis=3)(x)

        # Block 3
        x = Conv2D(filters=x.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_wise=True)(x)
        
        skip_connection = Input(shape=(32, 32, 3))
        x = keras.layers.Add()([x, skip_connection])
        
        # Fully connected layer
        x = Flatten()(x)
        output_layer = Dense(units=10, activation='softmax')(x)
        
        model = keras.Model(inputs=[input_layer, skip_connection], outputs=output_layer)

        return model