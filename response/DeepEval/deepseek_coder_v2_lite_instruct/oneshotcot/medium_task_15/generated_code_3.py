import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Concatenate, Multiply, AveragePooling2D, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(batch_norm1)
    
    # Two fully connected layers
    dense1 = Dense(units=64, activation='relu')(gap)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Reshape dense2 to match the number of channels in the initial features
    reshape_dense2 = Dense(units=32, activation='linear')(dense2)
    reshape_dense2 = keras.layers.Reshape((1, 1, 32))(reshape_dense2)
    
    # Multiply reshaped dense2 with the initial feature maps
    weighted_features = Multiply()([batch_norm1, reshape_dense2])
    
    # Concatenate weighted features with the original input
    concatenated = Concatenate()([input_layer, weighted_features])
    
    # 1x1 convolution and average pooling for downsampling
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    avg_pool = AveragePooling2D(pool_size=(8, 8), strides=1)(conv2)
    
    # Flatten and output layer
    flattened = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and return the model
model = dl_model()