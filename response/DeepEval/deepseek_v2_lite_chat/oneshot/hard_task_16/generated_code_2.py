import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, multiply, add, GlobalMaxPooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups for Block 1
    x = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    
    # Block 1
    x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x[0])
    x2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x[1])
    x3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x[2])
    
    # Concatenate the outputs from Block 1
    concat = Concatenate(axis=-1)([x1, x2, x3])
    
    # Transition Convolution
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(concat)
    
    # Block 2
    x = GlobalMaxPooling2D()(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dense(units=256, activation='relu')(x)
    x = Dense(units=128, activation='relu')(x)
    
    # Branch connection
    branch_output = Dense(units=10, activation='softmax')(x)
    
    # Combine main path and branch outputs
    output = add([concat, branch_output])
    
    # Final fully connected layer for classification
    output = Dense(units=10, activation='softmax')(output)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model

model = dl_model()
model.summary()