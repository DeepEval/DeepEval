import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, BatchNormalization, ReLU, Softmax

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolution
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation=ReLU(), padding='same')(input_layer)
    
    # First block
    batchnorm1 = BatchNormalization()(conv1)
    block1_output = ReLU()(batchnorm1)
    
    # Second block
    batchnorm2 = BatchNormalization()(conv1)
    block2_output = ReLU()(batchnorm2)
    
    # Third block
    batchnorm3 = BatchNormalization()(conv1)
    block3_output = ReLU()(batchnorm3)
    
    # Add outputs of the blocks to enhance feature representation
    add_layer = Add()([conv1, block1_output, block2_output, block3_output])
    
    # Flatten and fully connected layers for classification
    flatten = Flatten()(add_layer)
    dense1 = Dense(units=128, activation=ReLU(), kernel_initializer='he_uniform')(flatten)
    dense2 = Dense(units=64, activation=ReLU(), kernel_initializer='he_uniform')(dense1)
    output_layer = Dense(units=10, activation=Softmax(), kernel_initializer='glorot_uniform')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])