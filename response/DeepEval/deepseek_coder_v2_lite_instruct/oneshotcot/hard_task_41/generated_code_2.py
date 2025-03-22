import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Dropout, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block1(input_tensor):
        # Three parallel paths with different pooling scales
        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_tensor)
        
        # Flatten each path
        path1_flat = Flatten()(path1)
        path2_flat = Flatten()(path2)
        path3_flat = Flatten()(path3)
        
        # Concatenate the flattened paths
        concatenated = Concatenate()([path1_flat, path2_flat, path3_flat])
        
        # Dropout regularization
        dropout_output = Dropout(0.5)(concatenated)
        
        return dropout_output

    block1_output = block1(input_layer)

    # Fully connected layer and reshape to prepare for Block 2
    dense_layer = Dense(units=128, activation='relu')(block1_output)
    reshape_layer = Reshape((4, 4, 8))(dense_layer)  # Assuming 8 units in the dense layer

    # Block 2
    def block2(input_tensor):
        # Four branches
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        
        branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        
        branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)
        
        # Concatenate the outputs of the branches
        concatenated_output = Concatenate()([branch1, branch2, branch3, branch4])
        
        return concatenated_output

    block2_output = block2(reshape_layer)

    # Flatten the output of Block 2
    flatten_layer = Flatten()(block2_output)

    # Two fully connected layers for final classification
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.summary()