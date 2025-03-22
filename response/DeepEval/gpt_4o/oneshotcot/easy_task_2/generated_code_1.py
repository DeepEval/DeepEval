import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout, Softmax

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(224, 224, 3))
    
    # Step 2: Add first feature extraction layer (conv + avg pooling)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    
    # Step 3: Add second feature extraction layer (conv + avg pooling)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(avg_pool1)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    
    # Step 4: Add three additional convolutional layers
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(avg_pool2)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv4)
    
    # Step 5: Add another average pooling layer
    avg_pool3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv5)
    
    # Step 6: Flatten the feature maps
    flatten_layer = Flatten()(avg_pool3)
    
    # Step 7: Add fully connected layer with dropout
    dense1 = Dense(units=1024, activation='relu')(flatten_layer)
    dropout1 = Dropout(rate=0.5)(dense1)
    
    # Step 8: Add another fully connected layer with dropout
    dense2 = Dense(units=512, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.5)(dense2)
    
    # Step 9: Add output layer with softmax activation
    output_layer = Dense(units=1000, activation='softmax')(dropout2)
    
    # Step 10: Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model