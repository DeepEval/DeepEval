import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Concatenate, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch1)
    branch1 = Dropout(rate=0.5)(branch1)
    
    # Branch 2
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch2)
    branch2 = Dropout(rate=0.5)(branch2)
    
    # Branch 3
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch3)
    branch3 = Dropout(rate=0.5)(branch3)
    
    # Concatenate and process through FC layers
    concat = Concatenate()( [branch1, branch2, branch3] )
    fc1 = Dense(units=128, activation='relu')(concat)
    fc2 = Dense(units=64, activation='relu')(fc1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(fc2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Model instantiation and compilation
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])