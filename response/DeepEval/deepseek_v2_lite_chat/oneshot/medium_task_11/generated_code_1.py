import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Dense, Activation, BatchNormalization, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv1 = Conv2D(32, kernel_size=(3, 3), padding='same')(input_layer)
    batchnorm1 = BatchNormalization()(conv1)
    
    # Path 1: Global Average Pooling
    avg_pool1 = GlobalAveragePooling2D()(batchnorm1)
    dense1_avg = Dense(units=128, activation='relu')(avg_pool1)
    dense1_avg = Dense(units=64, activation='relu')(dense1_avg)
    
    # Path 2: Global Max Pooling
    max_pool1 = GlobalMaxPooling2D()(batchnorm1)
    dense2_max = Dense(units=128, activation='relu')(max_pool1)
    dense2_max = Dense(units=64, activation='relu')(dense2_max)
    
    # Concatenate the outputs from both paths
    concat = Concatenate()([dense1_avg, dense2_max])
    
    # Activation for channel attention weights
    attention_weights = Activation('sigmoid')(concat)
    
    # Element-wise multiplication with original features
    attention_conv1 = keras.layers.multiply([batchnorm1, attention_weights])
    
    # Path 3: Average Pooling
    avg_pool2 = AveragePooling2D(pool_size=(4, 4))(attention_conv1)
    dense3_avg = Dense(units=64, activation='relu')(avg_pool2)
    
    # Path 4: Max Pooling
    max_pool2 = MaxPooling2D(pool_size=(4, 4))(attention_conv1)
    dense4_max = Dense(units=64, activation='relu')(max_pool2)
    
    # Concatenate the outputs from both average and max pooling paths
    concat2 = Concatenate()([dense3_avg, dense4_max])
    
    # Fully connected layer with output size = 10
    dense5 = Dense(units=10)(concat2)
    output_layer = Activation('softmax')(dense5)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()