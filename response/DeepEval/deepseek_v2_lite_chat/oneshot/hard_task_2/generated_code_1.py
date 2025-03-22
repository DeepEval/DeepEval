import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the channel dimension
    group1 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(inputs)
    group2 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(inputs)
    group3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(inputs)
    
    # Convolution layers
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group1[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(group2[1])
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group3[2])
    
    # Pooling layers
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Concatenate the outputs from the three groups
    main_path = Concatenate()([pool1, pool2, pool3])
    
    # Add the original input layer to the main path
    combined_features = Concatenate()([main_path, inputs])
    
    # Flatten the combined features
    flatten = Flatten()(combined_features)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=inputs, outputs=output)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])