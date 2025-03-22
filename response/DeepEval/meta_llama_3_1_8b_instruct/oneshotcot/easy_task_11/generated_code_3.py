import keras
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # Apply average pooling layer for feature dimensionality reduction
    avg_pooling = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(input_layer)
    
    # Apply 1x1 convolutional layer to enhance depth and introduce nonlinearity
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pooling)
    
    # Flatten the feature maps
    flatten_layer = Flatten()(conv)
    
    # Apply the first fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Apply dropout layer to mitigate overfitting
    dropout = Dropout(0.2)(dense1)
    
    # Apply the second fully connected layer
    output_layer = Dense(units=10, activation='softmax')(dropout)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model