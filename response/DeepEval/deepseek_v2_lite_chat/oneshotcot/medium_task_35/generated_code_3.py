import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D, ZeroPadding2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Stage 1: Convolution and Max Pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Stage 2: Additional Convolution and Max Pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Stage 3: Multi-level Feature Extraction
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=2)(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor
    
    block_output = block(maxpool2)
    bath_norm = BatchNormalization()(block_output)
    flatten = Flatten()(bath_norm)
    
    # Stage 4: Skip Connections and UpSampling
    def skip_connection(input_tensor, num_filters):
        up = UpSampling2D(size=(2, 2))(input_tensor)
        zero_pad = ZeroPadding2D(padding=(1, 1))(up)
        conv = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', padding='same')(zero_pad)
        return conv
    
    skip_convs = [
        skip_connection(block_output, 32),
        skip_connection(block_output, 64),
        skip_connection(block_output, 128)
    ]
    
    conv_outputs = []
    for skip_conv, num_filters in zip(skip_convs, [32, 64, 128]):
        conv = Conv2D(num_filters, kernel_size=(3, 3), activation='relu')(skip_conv)
        conv_outputs.append(conv)
    
    conv_outputs = Concatenate()(conv_outputs)
    conv_outputs = Conv2D(10, kernel_size=(1, 1), activation='softmax')(conv_outputs)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=conv_outputs)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])