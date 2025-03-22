import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Process through three max pooling layers
    # Step 2: Add convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add maxpooling layer
    maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(conv1)
    
    # Step 4: Add another maxpooling layer with a larger window size
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    
    # Step 5: Add a third maxpooling layer with an even larger stride
    maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(conv1)
    
    # Flatten and concatenate the outputs
    flat1 = Flatten()(maxpool3)
    concat = Concatenate()([flat1, maxpool2, maxpool1])
    
    # Block 2: Multiple branches
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(concat)
    branch4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat)
    
    # Process each branch through max pooling and concatenate
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(branch1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch2)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=3, padding='valid')(branch3)
    pool4 = MaxPooling2D(pool_size=(3, 3), strides=3, padding='valid')(branch4)
    
    concat_branches = Concatenate()([pool1, pool2, pool3, pool4])
    
    # Flatten and pass through dense layers
    flatten = Flatten()(concat_branches)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()