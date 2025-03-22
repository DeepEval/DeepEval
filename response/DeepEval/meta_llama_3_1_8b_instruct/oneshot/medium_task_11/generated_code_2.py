import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Multiply, Add
from keras.regularizers import l2

def dl_model():
    
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the channel attention block
    def channel_attention(input_tensor):
        
        # Global average pooling
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        avg_fc1 = Dense(units=128, activation='relu', kernel_regularizer=l2(0.01))(avg_pool)
        avg_fc2 = Dense(units=128, activation='relu', kernel_regularizer=l2(0.01))(avg_fc1)
        avg_output = Dense(units=64, activation='sigmoid', kernel_regularizer=l2(0.01))(avg_fc2)

        # Global max pooling
        max_pool = GlobalMaxPooling2D()(input_tensor)
        max_fc1 = Dense(units=128, activation='relu', kernel_regularizer=l2(0.01))(max_pool)
        max_fc2 = Dense(units=128, activation='relu', kernel_regularizer=l2(0.01))(max_fc1)
        max_output = Dense(units=64, activation='sigmoid', kernel_regularizer=l2(0.01))(max_fc2)

        # Add the outputs and apply the sigmoid function
        add_output = Add()([avg_output, max_output])
        output = Dense(units=64, activation='sigmoid', kernel_regularizer=l2(0.01))(add_output)

        return Multiply()([input_tensor, output])

    # Apply channel attention
    channel_attention_output = channel_attention(conv)

    # Define the spatial attention block
    def spatial_attention(input_tensor):
        
        # Average pooling
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        avg_conv = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(avg_pool)

        # Max pooling
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        max_conv = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(max_pool)

        # Concatenate the outputs
        concat_output = Concatenate()([avg_conv, max_conv])

        # Apply element-wise multiplication
        return Multiply()([input_tensor, concat_output])

    # Apply spatial attention
    spatial_attention_output = spatial_attention(channel_attention_output)

    # Concatenate the spatial and channel attention outputs
    concat_output = Concatenate()([channel_attention_output, spatial_attention_output])

    # Apply max pooling
    max_pool_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(concat_output)

    # Flatten the output
    flatten_layer = Flatten()(max_pool_output)

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model