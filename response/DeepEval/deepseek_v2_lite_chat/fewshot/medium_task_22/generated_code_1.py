import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense, Multiply

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 Convolution
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    # Branch 2: 1x1 Convolution + 3x3 Convolution
    conv2 = Conv2D(filters=32, kernel_size=(1, 1))(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv2)
    # Branch 3: Max Pooling
    avg_pool = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    
    # Concatenate feature maps from different branches
    concat = Concatenate(axis=-1)([conv1, conv2, avg_pool])
    
    # Flatten the concatenated feature maps
    flatten = Flatten()(concat)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

return model