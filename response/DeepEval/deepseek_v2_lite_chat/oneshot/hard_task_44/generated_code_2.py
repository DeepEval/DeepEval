import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    # Split input into three groups
    group1, group2, group3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    
    # Convolutional layers for each group
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group2)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(group3)
    
    # Dropout layer
    conv1 = Dropout(0.2)(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv3 = Dropout(0.2)(conv3)
    
    # Concatenate outputs from different groups
    concat_layer = Concatenate(axis=-1)([conv1, conv2, conv3])
    
    # Block 2
    def branch(input_tensor, filters, size):
        conv = Conv2D(filters, size, padding='same', activation='relu')(input_tensor)
        return conv
    
    # Four branches
    branch1 = branch(input_tensor=concat_layer, filters=64, size=(1, 1))
    branch2 = branch(input_tensor=concat_layer, filters=64, size=(3, 3))
    branch3 = branch(input_tensor=concat_layer, filters=64, size=(5, 5))
    branch4 = branch(input_tensor=concat_layer, filters=64, size=(3, 3), max_pool=True)
    
    # Concatenate outputs from different branches
    output_branch1 = branch1
    output_branch2 = branch2
    output_branch3 = branch3
    output_branch4 = branch4
    
    output_branch1 = Flatten()(output_branch1)
    output_branch2 = Flatten()(output_branch2)
    output_branch3 = Flatten()(output_branch3)
    output_branch4 = Flatten()(output_branch4)
    
    output_branch = Concatenate()([output_branch1, output_branch2, output_branch3, output_branch4])
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(output_branch)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])