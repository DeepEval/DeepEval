import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, ZeroPadding2D, Conv2DTranspose, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        # 1x1 convolutional layer
        conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        # 3x3 convolutional layer
        conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_tensor)
        # Max pooling layer
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_tensor)
        # Up-sampling and 3x3 convolutional layer
        up1 = UpSampling2D(size=(2, 2))(pool1)
        conv3x3_up = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(up1)
        # Up-sampling and 1x1 convolutional layer
        up2 = UpSampling2D(size=(2, 2))(conv3x3_up)
        conv1x1_up = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(up2)
        # Concatenation of the outputs from all branches and another 1x1 convolutional layer
        concat = Concatenate()([conv1x1, conv3x3, conv1x1_up])
        conv1x1_final = Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(concat)
        
        return conv1x1_final
    
    # Branch path
    def branch_path(input_tensor):
        # 1x1 convolutional layer
        conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        # Max pooling layer
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1x1)
        # 3x3 convolutional layer
        conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool1)
        # Up-sampling and 1x1 convolutional layer
        up1 = UpSampling2D(size=(2, 2))(conv3x3)
        conv1x1_up = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(up1)
        # Concatenation of the outputs from both paths
        concat = Concatenate()([conv1x1_final, conv1x1_up])
        
        return concat
    
    # Construct the main path and branch path
    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)
    
    # Concatenate the outputs from both paths
    concat_layer = Concatenate()([main_output, branch_output])
    
    # Pass through two fully connected layers for classification
    dense1 = Dense(units=512, activation='relu')(concat_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model

# Instantiate and return the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()