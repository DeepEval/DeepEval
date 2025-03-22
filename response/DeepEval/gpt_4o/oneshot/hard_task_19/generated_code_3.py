import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path: three 3x3 convolutional layers followed by a max pooling layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
    
    # Branch path: global average pooling and channel weighting
    gap = GlobalAveragePooling2D()(conv3)
    dense1 = Dense(units=64, activation='relu')(gap)
    dense2 = Dense(units=128, activation='sigmoid')(dense1)
    
    # Reshape and multiply to apply channel weights
    weights = Reshape((1, 1, 128))(dense2)
    scaled_features = Multiply()([conv3, weights])
    
    # Combine the outputs from both paths
    combined = Add()([max_pool, scaled_features])
    
    # Classification: two fully connected layers
    flatten_layer = Flatten()(combined)
    dense3 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model