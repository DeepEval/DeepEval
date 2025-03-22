import keras
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add average pooling layer
    avg_pooling = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(input_layer)
    
    # Step 3: Add 1x1 convolutional layer
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pooling)
    
    # Step 4: Flatten the output
    flatten_layer = Flatten()(conv_1x1)
    
    # Step 5: Add first fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 6: Add dropout layer to mitigate overfitting
    dropout_layer = Dropout(rate=0.5)(dense1)
    
    # Step 7: Add second fully connected layer
    dense2 = Dense(units=64, activation='relu')(dropout_layer)
    
    # Step 8: Add output layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model