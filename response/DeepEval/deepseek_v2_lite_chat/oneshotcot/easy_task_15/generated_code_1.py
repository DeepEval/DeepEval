import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1)
    block1 = Dropout(0.2)(block1)  # Dropout layer to mitigate overfitting
    
    # Block 2
    block2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(block2)
    block2 = Dropout(0.2)(block2)  # Dropout layer to mitigate overfitting
    
    # Concatenate the outputs of the two blocks
    concat = Concatenate()(list(Block1=block1, Block2=block2))
    
    # Batch normalization and flattening
    bn = BatchNormalization()(concat)
    flatten = Flatten()(bn)
    
    # Fully connected layers
    fc1 = Dense(units=256, activation='relu')(flatten)
    fc2 = Dense(units=128, activation='relu')(fc1)
    output_layer = Dense(units=10, activation='softmax')(fc2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])