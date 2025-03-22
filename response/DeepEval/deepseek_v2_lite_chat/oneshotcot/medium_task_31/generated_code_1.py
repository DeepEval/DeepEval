import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split1 = Lambda(lambda x: tf.split(x, 3, axis=1))(inputs)
    split2 = Lambda(lambda x: tf.split(x, 3, axis=1))(inputs)
    split3 = Lambda(lambda x: tf.split(x, 3, axis=1))(inputs)
    
    # Convolutional layers
    conv1 = split1[0]  # 1x1 convolution
    conv2 = split2[0]  # 3x3 convolution
    conv3 = split3[0]  # 5x5 convolution
    
    # Pooling layers
    pool1 = split1[1]  # MaxPooling2D with a 3x3 window and a 2 stride
    pool2 = split2[1]  # MaxPooling2D with a 3x3 window and a 2 stride
    pool3 = split3[1]  # MaxPooling2D with a 3x3 window and a 2 stride
    
    # Concatenate the outputs from different scales
    concat = Concatenate()(list(conv1[0]) + list(conv2[0]) + list(conv3[0]))
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat)
    flat = Flatten()(batch_norm)
    
    # Dense layers
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(10, activation='softmax')(dense2)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=output)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])