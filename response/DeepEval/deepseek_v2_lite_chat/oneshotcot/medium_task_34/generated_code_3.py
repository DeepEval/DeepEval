import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, BatchNormalization, Flatten, Dense, Conv2DTranspose, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Extracting deep features through 3 pairs of <convolutional layer, max-pooling layer>
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Processing through <convolutional layer, Dropout layer, convolutional layer>
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool2)
    drop1 = Dropout(rate=0.2)(conv3)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(drop1)
    
    # Upsampling through 3 pairs of <convolutional layer, transposed convolutional layer>
    up1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(conv4)
    merge1 = Concatenate()([up1, conv3])
    up2 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(merge1)
    merge2 = Concatenate()([up2, conv2])
    up3 = Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(2, 2), activation='sigmoid')(merge2)
    
    # Skip connections for spatial information restoration
    skip1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    skip2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv4)
    merge = Add()([up3, skip2])
    
    # Final classification layer
    dense1 = Conv2D(filters=512, kernel_size=(1, 1), activation='relu')(merge)
    dense2 = Dense(units=1024, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()