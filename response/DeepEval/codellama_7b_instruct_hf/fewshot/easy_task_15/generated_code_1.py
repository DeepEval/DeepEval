import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Specialized block
    conv_3x3 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv_1x1 = Conv2D(32, (1, 1), activation='relu')(conv_3x3)
    pool = MaxPooling2D((2, 2))(conv_1x1)
    dropout = Dropout(0.2)(pool)
    
    # Global average pooling layer
    avg_pool = GlobalAveragePooling2D()(dropout)
    
    # Flatten layer
    flatten = Flatten()(avg_pool)
    
    # Fully connected layer
    output_layer = Dense(10, activation='softmax')(flatten)
    
    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model