import keras
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Average Pooling Layer
    avg_pool = AveragePooling2D(pool_size=(5, 5), strides=3)(input_layer)
    
    # 1x1 Convolutional Layer
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(avg_pool)
    
    # Flatten the feature maps
    flatten_layer = Flatten()(conv_1x1)
    
    # First Fully Connected Layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Dropout Layer
    dropout = Dropout(0.5)(dense1)
    
    # Second Fully Connected Layer
    dense2 = Dense(units=10, activation='softmax')(dropout)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model