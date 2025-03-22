import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Branch path
    branch_input = input_layer
    branch_output = branch_input

    # Main path: Global Average Pooling and Fully Connected Layers
    avg_pool = GlobalAveragePooling2D()(maxpool1)
    fc1 = Dense(units=128, activation='relu')(avg_pool)
    fc2 = Dense(units=64, activation='relu')(fc1)
    
    # Concatenate branch path and main path outputs
    concat = Concatenate()([branch_output, fc2])
    
    # Fully connected layers
    fc3 = Dense(units=128, activation='relu')(concat)
    fc4 = Dense(units=64, activation='relu')(fc3)
    output = Dense(units=10, activation='softmax')(fc4)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model

# Build the model
model = dl_model()
model.summary()