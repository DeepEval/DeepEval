import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, SeparableConv2D, BatchNormalization, ReLU
from keras.models import Model

def feature_extraction_block(input_tensor):
    # Main feature extraction block
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Concatenate the input tensor and the feature map
    x = Concatenate()([input_tensor, x])
    
    return x

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    main_path = feature_extraction_block(input_layer)
    main_path = feature_extraction_block(main_path)
    main_path = feature_extraction_block(main_path)

    # Branch path
    branch_path = Conv2D(filters=main_path.shape[-1], kernel_size=(1, 1), padding='same')(main_path)

    # Feature fusion
    fused_output = Add()([main_path, branch_path])
    
    # Flatten and output layers
    flatten_layer = Flatten()(fused_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model