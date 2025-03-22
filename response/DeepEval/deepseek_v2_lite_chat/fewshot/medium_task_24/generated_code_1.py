import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: <1x1 convolution, 3x3 convolution>
    branch1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    branch1_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch1_1)
    
    # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
    branch2_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    branch2_2 = Conv2D(filters=32, kernel_size=(1, 7), padding='same')(branch2_1)
    branch2_3 = Conv2D(filters=64, kernel_size=(7, 1), padding='same')(branch2_2)
    branch2_4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch2_3)
    
    # Branch 3: Max Pooling
    branch3 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    
    # Apply dropout to mitigate overfitting
    branch1_dropout = Dropout(rate=0.5)(branch1_2)
    branch2_dropout = Dropout(rate=0.5)(branch2_4)
    branch3_dropout = Dropout(rate=0.5)(branch3)
    
    # Concatenate all branches
    concat = Concatenate()([branch1_dropout, branch2_dropout, branch3_dropout])
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])