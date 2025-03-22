import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 1: Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)
    
    # Step 2: Global average pooling and dense layers to create channel weights
    global_avg_pool = GlobalAveragePooling2D()(relu)
    dense1 = Dense(units=32 // 2, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='sigmoid')(dense1)
    
    # Step 3: Reshape and multiply to generate weighted feature maps
    channel_weights = Reshape((1, 1, 32))(dense2)
    weighted_features = Multiply()([relu, channel_weights])
    
    # Step 4: Concatenate input layer and weighted features
    concatenated = Concatenate()([input_layer, weighted_features])
    
    # Step 5: 1x1 convolution to reduce dimensionality
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(concatenated)
    
    # Step 6: Average pooling
    avg_pooling = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1x1)
    
    # Step 7: Flatten and fully connected layer for classification
    flatten = Flatten()(avg_pooling)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model