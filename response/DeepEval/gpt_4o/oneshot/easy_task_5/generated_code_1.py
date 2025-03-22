import keras
from keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 Convolution for Dimensionality Reduction
    reduced_dim = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # 3x3 Convolution for Feature Extraction
    feature_extraction = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(reduced_dim)
    
    # 1x1 Convolution for Dimensionality Restoration
    restored_dim = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(feature_extraction)
    
    # Flatten the Output
    flatten_layer = Flatten()(restored_dim)
    
    # Fully Connected Layer for Classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model