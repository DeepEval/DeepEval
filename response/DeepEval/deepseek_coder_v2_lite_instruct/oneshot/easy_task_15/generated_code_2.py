import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor, filters):
        conv3x3 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1x1_1 = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)
        conv1x1_2 = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1_1)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1x1_2)
        dropout = Dropout(0.5)(avg_pool)
        return dropout
    
    # First block
    block1_output = block(input_tensor=input_layer, filters=32)
    
    # Second block
    block2_output = block(input_tensor=block1_output, filters=64)
    
    # Global Average Pooling
    global_avg_pool = keras.layers.GlobalAveragePooling2D()(block2_output)
    
    # Flatten
    flatten_layer = Flatten()(global_avg_pool)
    
    # Fully connected layer
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    
    return model