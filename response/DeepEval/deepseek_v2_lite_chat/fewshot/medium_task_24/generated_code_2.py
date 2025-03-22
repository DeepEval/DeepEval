import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution, 3x3 convolution, and max pooling
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2))(branch1)
    branch1 = Dropout(rate=0.3)(branch1)
    
    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2))(branch2)
    branch2 = Dropout(rate=0.3)(branch2)
    
    # Branch 3: max pooling
    branch3 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    branch3 = Dropout(rate=0.3)(branch3)
    
    # Concatenate and process through fully connected layers
    concatenated = Concatenate()([branch1, branch2, branch3])
    flatten = Flatten()(concatenated)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])