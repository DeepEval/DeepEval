import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv2_1)
    
    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    conv3_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv3_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv3_1)
    conv3_3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv3_2)
    
    # Branch 4: Average pooling
    avg_pool = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(input_layer)
    conv4_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(avg_pool)
    
    # Apply dropout for regularization
    conv1_dropout = Dropout(rate=0.5)(conv1_1)
    conv2_dropout = Dropout(rate=0.5)(conv2_2)
    conv3_dropout = Dropout(rate=0.5)(conv3_3)
    conv4_dropout = Dropout(rate=0.5)(conv4_1)
    
    # Concatenate the outputs from all branches
    concat = Concatenate(axis=-1)([conv1_dropout, conv2_dropout, conv3_dropout, conv4_dropout])
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and return the constructed model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()