import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Dropout, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the channel
    split_layer = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    
    # Feature extraction for each group with different convolutional layers
    def feature_extraction(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same')(conv2)
        dropout1 = Dropout(rate=0.5)(conv3)
        
        return dropout1
    
    # Apply feature extraction to each group
    group1 = feature_extraction(split_layer[0])
    group2 = feature_extraction(split_layer[1])
    group3 = feature_extraction(split_layer[2])
    
    # Concatenate the outputs from the three groups
    concat_layer = Concatenate(axis=-1)([group1, group2, group3])
    
    # Block 2: Separate branches for processing
    branch1 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(concat_layer)
    branch2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(concat_layer)
    branch3 = Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu')(concat_layer)
    branch4 = MaxPooling2D(pool_size=(1, 1), padding='same')(concat_layer)
    
    # Concatenate the outputs from all branches
    output_layer = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(output_layer)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and return the constructed model
model = dl_model()