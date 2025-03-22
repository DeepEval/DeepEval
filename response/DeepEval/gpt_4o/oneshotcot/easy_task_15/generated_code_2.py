import keras
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def specialized_block(input_tensor):
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv3x3)
        conv1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv1x1_1)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1x1_2)
        dropout_layer = Dropout(rate=0.5)(avg_pool)
        
        return dropout_layer

    # Apply the first block
    block1_output = specialized_block(input_layer)
    # Apply the second block
    block2_output = specialized_block(block1_output)
    
    # Global average pooling layer
    global_avg_pool = GlobalAveragePooling2D()(block2_output)
    
    # Flatten layer
    flatten_layer = Flatten()(global_avg_pool)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model