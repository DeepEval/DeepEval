import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, Concatenate, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path1_flat = Flatten()(path1)
    path1_drop = Dropout(0.5)(path1_flat)
    
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path2_flat = Flatten()(path2)
    path2_drop = Dropout(0.5)(path2_flat)
    
    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    path3_flat = Flatten()(path3)
    path3_drop = Dropout(0.5)(path3_flat)
    
    block1_output = Concatenate()([path1_drop, path2_drop, path3_drop])
    
    # Fully connected layer and reshaping operation
    fc1 = Dense(units=1024, activation='relu')(block1_output)
    reshape_layer = Reshape((4, 4, 64))(fc1)  # Example reshape, adjust dimensions if necessary
    
    # Block 2
    path1_b2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(reshape_layer)
    
    path2_b2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(reshape_layer)
    path2_b2_conv2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(path2_b2_conv1)
    path2_b2_conv3 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(path2_b2_conv2)
    
    path3_b2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(reshape_layer)
    path3_b2_conv2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(path3_b2_conv1)
    path3_b2_conv3 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(path3_b2_conv2)
    path3_b2_conv4 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(path3_b2_conv3)
    path3_b2_conv5 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(path3_b2_conv4)
    
    path4_b2_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(reshape_layer)
    path4_b2_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path4_b2_pool)
    
    block2_output = Concatenate()([path1_b2, path2_b2_conv3, path3_b2_conv5, path4_b2_conv])
    
    # Final fully connected layers for classification
    flatten_b2 = Flatten()(block2_output)
    dense_final1 = Dense(units=128, activation='relu')(flatten_b2)
    output_layer = Dense(units=10, activation='softmax')(dense_final1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model