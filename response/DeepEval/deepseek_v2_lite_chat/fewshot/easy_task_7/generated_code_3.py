import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Dropout

def dl_model():
    # Main path input
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv1 = Dropout(0.2)(conv1)  # Add dropout for regularization
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    conv2 = Dropout(0.2)(conv2)  # Add dropout for regularization
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Branch path directly connected to input
    branch_input = input_layer
    branch_conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch_input)
    branch_pool1 = MaxPooling2D(pool_size=(2, 2))(branch_conv1)
    
    # Combine outputs from both paths
    merged = Add()([pool2, branch_pool1])
    
    # Flattening layer
    flatten = Flatten()(merged)
    
    # Fully connected layer
    dense = Dense(units=128, activation='relu')(flatten)
    dropout = Dropout(0.5)(dense)  # Add dropout for regularization
    output_layer = Dense(units=10, activation='softmax')(dropout)  # 10 classes for MNIST
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])