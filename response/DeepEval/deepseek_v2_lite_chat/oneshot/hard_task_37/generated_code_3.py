import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch1_block = block(branch1)
    branch1_bn = BatchNormalization()(branch1_block)
    branch1_flatten = Flatten()(branch1_bn)
    branch1_dense1 = Dense(units=128, activation='relu')(branch1_flatten)
    branch1_dense2 = Dense(units=64, activation='relu')(branch1_dense1)
    
    # Branch 2
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch2_block = block(branch2)
    branch2_bn = BatchNormalization()(branch2_block)
    branch2_flatten = Flatten()(branch2_bn)
    branch2_dense1 = Dense(units=128, activation='relu')(branch2_flatten)
    branch2_dense2 = Dense(units=64, activation='relu')(branch2_dense1)
    
    # Combine branches
    combined_output = Add()([branch1_dense2, branch2_dense2])
    concat_layer = BatchNormalization()(combined_output)
    flatten_layer = Flatten()(concat_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Model architecture
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Define the block function
def block(input_tensor):
    path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path1)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(path2)
    max_pooling = MaxPooling2D(pool_size=(1, 1), padding='valid')(path3)
    output_tensor = Concatenate()([path1, path2, path3, max_pooling])
    
    return output_tensor

# Build and return the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])