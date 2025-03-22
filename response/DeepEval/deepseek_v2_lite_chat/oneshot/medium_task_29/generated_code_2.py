import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layer with 32 filters, each of size 3x3
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    
    # Pooling layer with a 1x1 window and stride of 1
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1)(conv1)
    
    # Pooling layer with a 2x2 window and stride of 2
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
    
    # Pooling layer with a 4x4 window and stride of 4
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4)(conv1)
    
    # Flatten the output and concatenate the results from different pooling layers
    flat_layer1 = Flatten()(pool1)
    flat_layer2 = Flatten()(pool2)
    flat_layer3 = Flatten()(pool3)
    
    # Concatenate the flattened outputs
    concat_layer = Concatenate()([flat_layer1, flat_layer2, flat_layer3])
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(concat_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and return the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])