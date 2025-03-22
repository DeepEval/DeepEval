import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Multiply

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups for each convolution
    split1 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    split2 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    split3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    
    # Convolution layers
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split1[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split2[1])
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split3[2])
    
    # Concatenate the outputs from the three paths
    concat = Concatenate(axis=-1)([conv1, conv2, conv3])
    
    # Transition layer to adjust channel count
    transition = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concat)
    
    # Block 2
    pool = MaxPooling2D(pool_size=(2, 2))(transition)
    reshape = Reshape((-1, 64))(pool)  # Reshape for fully connected layers
    
    # Fully connected layers for generating weights
    fc1 = Dense(units=512, activation='relu')(reshape)
    fc2 = Dense(units=256, activation='relu')(fc1)
    
    # Generate weights for reshaping the transition layer
    weights = Dense(units=64 * 4)(fc2)
    weights = keras.backend.reshape(weights, (-1, 4, 16))  # Reshape back to (batch_size, 4, 16)
    
    # Multiply the weights with the transition layer to generate main path output
    main_output = Multiply()([weights, transition])
    
    # Branch output
    branch = Dense(units=10, activation='softmax')(fc1)
    
    # Combine main path and branch outputs
    combined_output = keras.layers.Add()([main_output, branch])
    
    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(combined_output)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()