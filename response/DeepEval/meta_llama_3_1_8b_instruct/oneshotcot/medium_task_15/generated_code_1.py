import keras
from keras.layers import Input, Conv2D, BatchNormalization, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D
from keras.models import Model

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Step 1: Convolutional layer
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 2: Batch Normalization layer
    bath_norm = BatchNormalization()(conv)
    
    # Step 3: Global Average Pooling layer
    global_pooling = GlobalAveragePooling2D()(bath_norm)
    
    # Step 4: Fully connected layer
    dense1 = Dense(units=64, activation='relu')(global_pooling)
    
    # Step 5: Reshape the output to match the size of the initial feature
    reshape = Reshape((64,))(dense1)
    
    # Step 6: Multiply the output with the initial feature
    multiply = Multiply()([reshape, bath_norm])
    
    # Step 7: Concatenate the result with the input layer
    concat = Concatenate()([multiply, input_layer])
    
    # Step 8: 1x1 convolution layer
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat)
    
    # Step 9: Average Pooling layer
    avg_pooling = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1x1)
    
    # Step 10: Flatten the result
    flatten_layer = Flatten()(avg_pooling)
    
    # Step 11: Final fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model