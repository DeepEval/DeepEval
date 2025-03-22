import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Initial Convolutional Layer
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='sigmoid')(input_layer)
    
    # Splitting the feature maps into two groups
    split_1 = Lambda(lambda x: keras.backend.split(x, 2, axis=-1))(conv_layer)
    split_2 = Lambda(lambda x: keras.backend.split(x, 2, axis=-1))(conv_layer)
    
    # Operations for Group 1
    def block1_operations(group):
        group_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group[0])
        group_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(group[0])
        group_3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group[1])
        return Concatenate()([group_1, group_2, group_3])
    
    group_1 = block1_operations(split_1)
    group_2 = block1_operations(split_2)
    
    # Merging the outputs from both groups
    merge_output = Concatenate()([group_1, group_2])
    
    # Block 2: Channel Shuffling and Fully Connected Layer
    reshape_input = Lambda(lambda x: keras.backend.reshape(x, (-1, x.shape[1]*x.shape[2])))(merge_output)
    dense1 = Dense(128, activation='relu')(reshape_input)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()