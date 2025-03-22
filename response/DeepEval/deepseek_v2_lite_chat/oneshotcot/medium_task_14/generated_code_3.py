import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Concatenate, Dense, Flatten

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Block 1
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
    
    # Block 2
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
    
    # Block 3
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)
    
    # Parallel branch
    conv4 = Conv2D(64, (1, 1), activation='relu')(inputs)
    
    # Split blocks into separate paths
    path1 = Flatten()(bn1)
    path2 = Flatten()(bn2)
    path3 = Flatten()(bn3)
    path4 = Flatten()(conv4)
    
    # Concatenate paths
    concat = Concatenate()([path1, path2, path3, path4])
    
    # Fully connected layers
    dense1 = Dense(512, activation='relu')(concat)
    dense2 = Dense(256, activation='relu')(dense1)
    dense3 = Dense(128, activation='relu')(dense2)
    output = Dense(10, activation='softmax')(dense3)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=output)
    
    return model

# Instantiate the model
model = dl_model()

# Print model summary
model.summary()