import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    def custom_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
        dropout = Dropout(rate=0.3)(avg_pool)
        return dropout
    
    # Apply the custom block twice consecutively
    block1_output = custom_block(input_layer)
    block2_output = custom_block(block1_output)
    
    # Global Average Pooling layer to reduce dimensions
    global_avg_pooling = GlobalAveragePooling2D()(block2_output)
    
    # Flatten and Fully Connected layer
    flatten_layer = Flatten()(global_avg_pooling)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model