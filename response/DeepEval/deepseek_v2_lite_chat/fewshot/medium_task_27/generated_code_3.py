import keras
from keras.layers import Input, Conv2D, MaxPool2D, Add, AveragePooling2D, Flatten, Dense, Multiply, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 convolutional layer
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')(branch1)
    
    # Branch 2: 5x5 convolutional layer
    branch2 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    pool2 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')(branch2)
    
    # Add branches
    add_layer = Add()([pool1, pool2])
    
    # Average pooling layer
    avg_pool = AveragePooling2D(pool_size=(4, 4))(add_layer)
    
    # Flatten layer
    flatten = Flatten()(avg_pool)
    
    # Two fully connected layers with softmax for attention weights
    dense1 = Dense(units=1024, activation='relu')(flatten)
    dense2 = Dense(units=512, activation='relu')(dense1)
    
    # Attention weights
    attention_weights = Dense(units=2, activation='softmax', name='attention_weights')(dense2)
    
    # Multiply attention weights with branch outputs
    weighted_output1 = Multiply()([branch1, attention_weights])
    weighted_output2 = Multiply()([branch2, attention_weights])
    
    # Combine weighted outputs
    combined_output = Concatenate()([weighted_output1, weighted_output2])
    
    # Fully connected layer for final classification
    output_layer = Dense(units=10, activation='softmax')(combined_output)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()