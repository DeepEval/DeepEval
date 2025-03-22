import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups
    group1, group2, group3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    
    # Feature extraction for main path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(group1)
    conv1 = BatchNormalization()(conv1)
    conv1 = keras.activations.relu(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(group2)
    conv2 = BatchNormalization()(conv2)
    conv2 = keras.activations.relu(conv2)
    
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(group3)
    conv3 = BatchNormalization()(conv3)
    conv3 = keras.activations.relu(conv3)
    
    # Concatenate features from main path
    concat = Concatenate(axis=-1)([conv1, conv2, conv3])
    
    # Branch path
    branch = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    branch = BatchNormalization()(branch)
    branch = keras.activations.relu(branch)
    
    # Combine main path and branch path
    combined = keras.layers.Add()([concat, branch])
    
    # Additional 3x3 convolution
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(combined)
    conv4 = BatchNormalization()(conv4)
    conv4 = keras.activations.relu(conv4)
    
    # Max pooling
    pool = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Flatten and fully connected layers
    flatten = Flatten()(pool)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()
model.summary()