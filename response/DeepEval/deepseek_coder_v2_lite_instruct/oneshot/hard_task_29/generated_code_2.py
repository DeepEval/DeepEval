import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    def first_block(input_tensor):
        # Main path
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        
        # Branch path
        branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Combine both paths
        added = Add()([conv2, branch])
        return added
    
    block1_output = first_block(input_layer)
    
    # Second block with max pooling layers
    def second_block(input_tensor):
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        
        # Flatten the results
        flattened = Flatten()(pool1)
        flattened2 = Flatten()(pool2)
        flattened3 = Flatten()(pool3)
        
        # Concatenate the flattened outputs
        concatenated = Concatenate()([flattened, flattened2, flattened3])
        return concatenated
    
    block2_output = second_block(block1_output)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(block2_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()