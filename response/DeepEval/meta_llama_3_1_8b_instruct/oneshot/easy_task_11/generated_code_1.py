import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # Average pooling with 5x5 window and 3x3 stride for feature dimensionality reduction
    avg_pooling = AveragePooling2D(pool_size=(5, 5), strides=3, padding='valid')(input_layer)
    
    # 1x1 convolutional layer to enhance depth and introduce nonlinearity
    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pooling)
    
    # Flatten the feature maps
    flatten_layer = Flatten()(conv)
    
    # First fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Dropout layer to prevent overfitting
    dropout = Dropout(0.2)(dense1)
    
    # Second fully connected layer
    dense2 = Dense(units=64, activation='relu')(dropout)
    
    # Output layer with softmax activation for multi-class classification
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model