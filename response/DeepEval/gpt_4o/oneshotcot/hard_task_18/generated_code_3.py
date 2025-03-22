import keras
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    # Main path
    conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1_1)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), padding='same')(conv1_2)
    
    # Skip connection
    skip1 = Add()([input_layer, avg_pool1])
    
    # Second Block
    # Main path
    global_avg_pool = GlobalAveragePooling2D()(skip1)
    dense2_1 = Dense(units=32, activation='relu')(global_avg_pool)  # Assuming 32 channels from first block
    dense2_2 = Dense(units=64, activation='sigmoid')(dense2_1)  # Match the number of channels in the first block's output
    
    # Reshape and scale
    reshape = Reshape((1, 1, 64))(dense2_2)  # Reshape to match the feature map shape
    scaled_features = Multiply()([skip1, reshape])
    
    # Flatten and final Dense layer for classification
    flatten_layer = Flatten()(scaled_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model