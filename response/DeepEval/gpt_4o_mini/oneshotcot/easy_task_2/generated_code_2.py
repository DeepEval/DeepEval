import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(224, 224, 3))
    
    # Step 2: First feature extraction layer: Conv + Avg Pool
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    
    # Step 3: Second feature extraction layer: Conv + Avg Pool
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool1)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Step 4: Additional convolutional layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool2)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    
    # Step 5: Final average pooling layer
    avg_pool3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv5)

    # Step 6: Flatten the feature maps
    flatten_layer = Flatten()(avg_pool3)
    
    # Step 7: Fully connected layers with Dropout
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dropout1 = Dropout(0.5)(dense1)
    
    dense2 = Dense(units=256, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    
    # Step 8: Output layer for classification
    output_layer = Dense(units=1000, activation='softmax')(dropout2)

    # Step 9: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model