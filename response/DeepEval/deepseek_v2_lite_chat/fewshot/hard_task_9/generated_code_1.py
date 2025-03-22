import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, AveragePooling2D, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Feature extraction branches
    branch1 = Conv2D(filters=64, kernel_size=1, activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=1, activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=3, activation='relu')(branch2)
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=3, activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=3, activation='relu')(branch3)
    
    # Concatenate the outputs of the branches
    concat = Concatenate()([branch1, branch2, branch3])
    
    # Adjust output dimensions to match input channel size
    conv1x1 = Conv2D(filters=3, kernel_size=1, activation='sigmoid')(concat)
    
    # Main path with addition
    main_path = Add()([input_layer, conv1x1])
    
    # Batch normalization and flattening
    bn = BatchNormalization()(main_path)
    flatten = Flatten()(bn)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])