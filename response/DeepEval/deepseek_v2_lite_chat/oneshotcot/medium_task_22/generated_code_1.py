import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 Convolution
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    
    # Branch 2: 1x1 Convolution, followed by two 3x3 Convolutions
    conv2 = Conv2D(filters=32, kernel_size=(1, 1))(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv2)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv2)
    
    # Branch 3: Max Pooling
    maxpool = MaxPooling2D(pool_size=(2, 2))(input_layer)
    
    # Concatenate branches
    concat = Concatenate()([conv1, conv2, maxpool])
    
    # Batch Normalization and Flatten
    batch_norm = BatchNormalization()(concat)
    flatten = Flatten()(batch_norm)
    
    # Two Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])