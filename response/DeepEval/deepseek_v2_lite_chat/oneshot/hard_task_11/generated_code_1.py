import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Average

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main pathway
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Branch with 1x1, 1x3, and 3x1 convolutions
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(input_layer)

    # Concatenate the outputs of the parallel branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Additional 1x1 convolution to match the output dimensions of the main pathway
    conv_concat = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concatenated)
    
    # Additive connection with the main pathway
    fused = Average()[(conv1 + conv_concat)]
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(fused)
    flattened = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])