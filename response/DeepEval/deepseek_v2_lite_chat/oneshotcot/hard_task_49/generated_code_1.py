import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    block1_output = []
    for pooling_size in [1, 2, 4]:
        avg_pooling = AveragePooling2D(pool_size=(pooling_size, pooling_size), strides=(1, 1))(input_layer)
        block1_output.append(Flatten()(avg_pooling))
    
    # Concatenate the flattened outputs from each pooling layer
    block1_output = Concatenate()(block1_output)
    
    # Transform the concatenated vector into a 4D tensor for input into the second block
    block1_output = Dense(units=4 * block1_output.shape[1])(block1_output)
    block1_output = Lambda(lambda x: tf.split(x, [int(x.shape[0] / 4),]*4,axis=0))(block1_output)
    
    # Block 2
    block2_output = []
    for i in range(4):
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output[i])
        if i < 3:
            conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
        if i < 2:
            conv = Conv2D(filters=96, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv)
        if i < 1:
            conv = Conv2D(filters=128, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(conv)
        conv = BatchNormalization()(conv)
        block2_output.append(Flatten()(conv))
    
    # Concatenate the flattened outputs from each group of convolutions
    block2_output = Concatenate()(block2_output)
    
    # Pass through a fully connected layer to produce the classification result
    output_layer = Dense(units=10, activation='softmax')(block2_output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model