import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D, Concatenate
from keras.models import Model
from keras.layers import Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    block1 = []
    for pool_size, stride in [(1, 1), (2, 2), (4, 4)]:
        block1.append(AveragePooling2D(pool_size=pool_size, strides=stride)(input_layer))
    block1 = Flatten()(block1)
    block1 = Dropout(0.5)(block1)  # Dropout for regularization
    
    # Concatenate and process by a single dense layer
    concat_layer = Concatenate()(block1)
    dense1 = Dense(128, activation='relu')(concat_layer)
    dense1 = Dropout(0.5)(dense1)  # Dropout for regularization
    
    # Block 2
    branch1 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(64, (1, 1), activation='relu')(branch1)
    branch3 = Conv2D(64, (3, 3), activation='relu')(branch1)
    branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(branch1)
    
    # Concatenate and process each branch
    branch1 = Dense(64, activation='relu')(branch3)
    branch2 = Dense(64, activation='relu')(branch4)
    branch3 = Dense(64, activation='relu')(branch3)
    branch4 = Dense(64, activation='relu')(branch4)
    
    # Concatenate the outputs of branches
    fused_layer = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
    fused_layer = Dropout(0.5)(fused_layer)  # Dropout for regularization
    
    # Fully connected layers
    output = Dense(10, activation='softmax')(fused_layer)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Building the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])