import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Define the specialized block
    def feature_block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        x = Conv2D(filters=16, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = Conv2D(filters=8, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = AveragePooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(rate=0.25)(x)  
        return x

    # Apply the block twice
    block1_output = feature_block(input_layer)
    block2_output = feature_block(block1_output)

    # Global average pooling and flattening
    global_avg_pool = GlobalAveragePooling2D()(block2_output)
    flatten_layer = Flatten()(global_avg_pool)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model