from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, GlobalMaxPooling2D, Add, Activation, Reshape, Multiply, Concatenate, Flatten, AveragePooling2D, MaxPooling2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    input_channels = input_layer.shape[-1]
    initial_conv = Conv2D(filters=input_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    def block_1(input_tensor):

        avgpool = GlobalAveragePooling2D()(input_tensor)
        avg_out = Dense(units=input_channels // 3, activation='relu')(avgpool)
        avg_out = Dense(units=input_channels, activation='relu')(avg_out)

        maxpool = GlobalMaxPooling2D()(input_tensor)
        max_out = Dense(units=input_channels // 3, activation='relu')(maxpool)
        max_out = Dense(units=input_channels, activation='relu')(max_out)

        added = Add()([avg_out, max_out])
        act = Activation(activation='sigmoid')(added)
        reshaped = Reshape(target_shape=(1, 1, input_channels))(act)
        output_tensor = Multiply()([input_tensor, reshaped])

        return output_tensor
    
    def block_2(input_tensor):

        avgpool = AveragePooling2D(pool_size=(2,2), strides=(1, 1), padding='same')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding='same')(input_tensor)
        spatial = Concatenate(axis=-1)([avgpool, maxpool])
        
        spatial_out = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(spatial)
        output_tensor = Multiply()([input_tensor, spatial_out])

        return output_tensor

    block1_output = block_1(input_tensor=initial_conv)
    block2_output = block_2(input_tensor=block1_output)
    main_path = block2_output
    branch_path = Conv2D(filters=input_channels, kernel_size=(1, 1), strides=(1,1), padding='same')(input_layer)
    added = Add()([main_path, branch_path])

    flatten = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
