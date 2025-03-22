import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Activation

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    
    def feature_block(input_tensor):
      # Separate Convolution
      depthwise_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')(input_tensor) 
      
      # Concatenation
      output_tensor = Concatenate()([input_tensor, depthwise_conv])
      return output_tensor

    # Main Path
    main_path = input_layer
    for _ in range(3):
      main_path = feature_block(main_path)

    # Branch Path
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Feature Fusion
    merged_features = Add()([main_path, branch_path])

    # Flatten and Classification
    flatten_layer = Flatten()(merged_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model