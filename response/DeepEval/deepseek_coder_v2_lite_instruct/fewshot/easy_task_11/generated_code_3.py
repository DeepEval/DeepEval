import keras
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Average Pooling Layer
    avg_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(input_layer)
    
    # 1x1 Convolutional Layer
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(avg_pool)
    
    # Flatten the feature maps
    flatten_layer = Flatten()(conv_1x1)
    
    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    
    # Output Layer
    output_layer = Dense(units=10, activation='softmax')(dropout2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model