import keras
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Apply average pooling
    avg_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(input_layer)
    
    # 1x1 Convolutional layer
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(avg_pool)
    
    # Flatten the feature maps
    flatten = Flatten()(conv_1x1)
    
    # First fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten)
    
    # Dropout layer for regularization
    dropout = Dropout(rate=0.5)(dense1)
    
    # Second fully connected layer
    dense2 = Dense(units=64, activation='relu')(dropout)
    
    # Output layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model