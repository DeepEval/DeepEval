import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Concatenate
from keras.layers import Dropout, BatchNormalization

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Average pooling layer with a 5x5 window and a 3x3 stride
    avg_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(input_layer)
    
    # 1x1 convolutional layer to add depth and introduce nonlinearity
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(avg_pool)
    
    # Flatten the feature maps and pass through two fully connected layers
    flatten = Flatten()(conv1)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Dropout to mitigate overfitting
    dropout = Dropout(0.5)(dense1)
    
    # Final output layer with softmax activation for multi-class classification
    output_layer = Dense(units=10, activation='softmax')(dropout)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model