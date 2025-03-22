import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    def block1(input_tensor):
        # Three average pooling layers with varying scales
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_tensor)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_tensor)
        # Flatten and concatenate
        flat1 = Flatten()(pool1)
        flat2 = Flatten()(pool2)
        flat3 = Flatten()(pool3)
        concat = Concatenate()([flat1, flat2, flat3])
        
        # Transformation to 4D tensor for second block
        reshape = Lambda(lambda x: keras.backend.reshape(x, (-1, x.shape[1]*x.shape[2]*x.shape[3])))([concat])
        
        return reshape
    
    def block2(input_tensor):
        # Split input into four groups, process each, and concatenate
        group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        group3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        group4 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        concat = Concatenate(axis=-1)([group1, group2, group3, group4])
        
        # Flatten and pass through fully connected layer
        flatten = Flatten()(concat)
        dense = Dense(units=128, activation='relu')(flatten)
        
        return dense
    
    # First block
    input_layer = Input(shape=(28, 28, 1))
    first_block_output = block1(input_layer)
    
    # Second block
    second_block_output = block2(first_block_output)
    
    # Final dense layers for classification
    output_layer = Dense(units=10, activation='softmax')(second_block_output)
    
    # Construct model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and print model summary
model = dl_model()
model.summary()