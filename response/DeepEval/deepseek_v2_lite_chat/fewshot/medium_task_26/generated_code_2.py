import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Lambda

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # Channel compression with 1x1 convolution
    channel_layer = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Expand features with parallel convolutional layers
    conv1_1 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(channel_layer)
    conv1_2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(channel_layer)
    
    # Concatenate the results
    concat = Add()([conv1_1, conv1_2])
    
    # Flatten and fully connected layers for classification
    flatten = Flatten()(concat)
    dense1 = Dense(units=1024, activation='relu')(flatten)
    dense2 = Dense(units=512, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()