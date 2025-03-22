import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer with batch normalization and ReLU activation
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_layer)
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)
    
    # Global average pooling layer
    global_avg_pool = GlobalAveragePooling2D()(relu)
    
    # Two fully connected layers to compress the feature maps
    dense1 = Dense(units=16, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='sigmoid')(dense1)
    
    # Reshape and multiply to create weighted feature maps
    reshape = Reshape((1, 1, 32))(dense2)
    weighted_features = Multiply()([relu, reshape])
    
    # Concatenate weighted features with the input layer
    concat = Concatenate()([input_layer, weighted_features])
    
    # Dimensionality reduction and downsampling
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    avg_pooling = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1x1)
    
    # Flatten and fully connected layer for classification
    flatten = Flatten()(avg_pooling)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model