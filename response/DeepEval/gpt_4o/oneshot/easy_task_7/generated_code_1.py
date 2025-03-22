import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    # First <convolution, dropout> block
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Dropout(rate=0.25)(main_path)
    
    # Second <convolution, dropout> block
    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    main_path = Dropout(rate=0.25)(main_path)
    
    # Convolution to restore the number of channels
    main_path = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(main_path)
    
    # Branch path
    branch_path = input_layer  # Direct connection to the input
    
    # Combine paths using addition
    combined = Add()([main_path, branch_path])
    
    # Flatten and fully connected layer for final classification
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model