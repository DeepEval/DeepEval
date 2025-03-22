import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Add, Dense, GlobalAveragePooling2D, Multiply, Flatten

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))
    
    # First block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = AveragePooling2D((2, 2))(x)
    shortcut = x
    
    # Main path
    y = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    y = Conv2D(64, (3, 3), activation='relu', padding='same')(y)
    y = AveragePooling2D((2, 2))(y)
    
    # Addition of first block and main path
    x = Add()([shortcut, y])
    
    # Second block
    y = GlobalAveragePooling2D()(x)
    y = Dense(64, activation='relu')(y)
    y = Dense(64, activation='relu')(y)
    weights = Dense(64, activation='softmax')(y)
    
    # Reshape weights to match input shape
    reshaped_weights = tf.expand_dims(tf.expand_dims(weights, 1), 1)
    
    # Multiply weights with input
    x = Multiply()([x, reshaped_weights])
    
    # Flatten the output
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model