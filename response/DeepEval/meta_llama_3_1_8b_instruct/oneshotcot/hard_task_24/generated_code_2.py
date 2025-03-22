import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Add initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Step 3: Define a block
    def branch1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv1
    
    def branch2(input_tensor):
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        upsample = UpSampling2D(size=(2, 2))(max_pool)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(upsample)
        return conv2
    
    def branch3(input_tensor):
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        upsample = UpSampling2D(size=(2, 2))(max_pool)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(upsample)
        return conv3
    
    # Step 4: Add the outputs of all branches
    branch_output1 = branch1(conv)
    branch_output2 = branch2(conv)
    branch_output3 = branch3(conv)
    merged_output = Concatenate()([branch_output1, branch_output2, branch_output3])
    
    # Step 5: Add another 1x1 convolutional layer
    conv_final = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(merged_output)
    
    # Step 6: Add flatten layer
    flatten_layer = Flatten()(conv_final)
    
    # Step 7: Add dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 8: Add dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 9: Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model