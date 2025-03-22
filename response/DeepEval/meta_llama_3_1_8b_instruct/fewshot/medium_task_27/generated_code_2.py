import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply, Flatten
from keras.layers import Lambda, Reshape

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset has images of 32x32 pixels with 3 color channels
    
    # Create two parallel convolutional branches
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the outputs of the two branches using addition
    adding_layer = Add()([conv1, conv2])
    
    # Apply global average pooling to compress the features
    global_avg_pool = GlobalAveragePooling2D()(adding_layer)
    
    # Create two fully connected layers to generate attention weights
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    # Create two sets of weights, one for each branch
    weight1 = Multiply()([dense2, Lambda(lambda x: x * 0.5)(global_avg_pool)])
    weight2 = Multiply()([dense2, Lambda(lambda x: x * 0.5)(global_avg_pool)])
    
    # Add the weighted outputs of the two branches
    output_layer = Add()([weight1, weight2])
    
    # Final fully connected layer to produce the probability distribution
    final_dense = Dense(units=10, activation='softmax')(output_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=final_dense)
    
    return model