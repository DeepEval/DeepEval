import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # Cifar images are 32x32 and have 3 color channels
    
    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv2)
    
    # Convolutional layers with strides for downsampling
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), activation='relu')(conv3)
    conv5 = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), activation='relu')(conv4)
    
    # Pooling layers
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv5)
    
    # Feature extraction paths
    path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(pool1)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path1)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(path2)
    
    # Concatenate features from different paths
    concat = Concatenate()([path1, path2, path3])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)  # Cifar-10 has 10 classes
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model