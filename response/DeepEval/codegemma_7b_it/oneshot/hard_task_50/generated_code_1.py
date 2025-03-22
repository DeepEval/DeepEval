import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    max_pooling_1x1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    max_pooling_2x2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    max_pooling_4x4 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    
    max_pool_flatten_1x1 = Flatten()(max_pooling_1x1)
    max_pool_flatten_2x2 = Flatten()(max_pooling_2x2)
    max_pool_flatten_4x4 = Flatten()(max_pooling_4x4)
    
    # Apply dropout to mitigate overfitting
    max_pool_flatten_1x1 = Dropout(0.5)(max_pool_flatten_1x1)
    max_pool_flatten_2x2 = Dropout(0.5)(max_pool_flatten_2x2)
    max_pool_flatten_4x4 = Dropout(0.5)(max_pool_flatten_4x4)
    
    concat_max_pool = Concatenate()([max_pool_flatten_1x1, max_pool_flatten_2x2, max_pool_flatten_4x4])
    
    # Second block
    def block(input_tensor):
        shape = keras.backend.int_shape(input_tensor)
        input_shape_rank = len(shape)
        
        groups = 4
        input_shapes_per_group = shape[-1] // groups
        
        outputs = []
        for i in range(groups):
            x = Lambda(lambda z: z[..., i * input_shapes_per_group : (i + 1) * input_shapes_per_group])(input_tensor)
            x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
            x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
            x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x)
            x = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(x)
            outputs.append(x)
        
        return outputs
    
    block_outputs = block(concat_max_pool)
    
    concat_block_outputs = Concatenate()(block_outputs)
    
    reshape_block_outputs = Reshape((shape[1] * shape[2], shape[-1]))(concat_block_outputs)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(reshape_block_outputs)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()