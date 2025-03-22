import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras.layers import Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: <1x1 convolution, 3x3 convolution>
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv4 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv5 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    
    # Branch 3: max pooling
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    
    # Concatenate the outputs from all branches
    concat_output = Concatenate()([conv2, conv6, max_pool])
    
    # Apply batch normalization
    batch_norm = BatchNormalization()(concat_output)
    
    # Apply dropout to mitigate overfitting
    dropout = Dropout(0.2)(batch_norm)
    
    # Flatten the result
    flatten_layer = Flatten()(dropout)
    
    # Apply three fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model