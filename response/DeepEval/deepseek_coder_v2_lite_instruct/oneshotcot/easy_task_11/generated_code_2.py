import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add average pooling layer
    avg_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(input_layer)
    
    # Step 3: Add 1x1 convolutional layer
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(avg_pool)
    
    # Step 4: Flatten the feature maps
    flatten_layer = Flatten()(conv1x1)
    
    # Step 5: Add fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout = Dropout(0.5)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout)
    
    # Step 6: Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Step 7: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model