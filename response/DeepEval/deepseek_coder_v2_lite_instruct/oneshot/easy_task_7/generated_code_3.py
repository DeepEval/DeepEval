import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    def main_path(x):
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
        return x
    
    main_output = main_path(input_layer)
    
    # Branch path
    branch_output = input_layer
    
    # Combine paths
    combined_output = Add()([main_output, branch_output])
    
    # Flatten and fully connected layer
    flatten_layer = Flatten()(combined_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model