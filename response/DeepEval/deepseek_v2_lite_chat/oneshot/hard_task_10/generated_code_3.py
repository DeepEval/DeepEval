import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # Assuming input shape as 32x32 pixels with 3 color channels
    
    # Feature extraction path 1: 1x1 convolution
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Feature extraction path 2
    conv_1x7 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_7x1 = Conv2D(filters=64, kernel_size=(7, 1), strides=(7, 1), padding='same', activation='relu')(input_layer)
    
    # Concatenate the outputs of paths 1 and 2
    concatenated = Concatenate(axis=-1)([conv_1x1, conv_1x7, conv_7x1])
    
    # Additional branch connected directly to the input
    branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Merge the main path and the branch
    merged = Add()([concatenated, branch])
    
    # 1x1 convolution to align the output dimensions with the input image's channel
    conv_merge = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(merged)
    
    # Batch normalization
    batch_norm = BatchNormalization()(conv_merge)
    
    # Flatten the result
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers for classification
    dense1 = Dense(units=512, activation='relu')(flatten)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # Assuming 10 classes
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and return the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()