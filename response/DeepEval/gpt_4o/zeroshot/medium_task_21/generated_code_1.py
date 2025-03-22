from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch1 = Dropout(0.3)(branch1)
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = Dropout(0.3)(branch2)
    
    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = Dropout(0.3)(branch3)
    
    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch4 = Conv2D(32, (1, 1), activation='relu')(branch4)
    branch4 = Dropout(0.3)(branch4)
    
    # Concatenate all branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Flatten the concatenated output
    flat = Flatten()(concatenated)
    
    # Fully connected layers
    fc1 = Dense(256, activation='relu')(flat)
    fc2 = Dense(128, activation='relu')(fc1)
    fc3 = Dense(64, activation='relu')(fc2)
    
    # Output layer for classification (10 classes for CIFAR-10)
    output_layer = Dense(10, activation='softmax')(fc3)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = dl_model()
model.summary()