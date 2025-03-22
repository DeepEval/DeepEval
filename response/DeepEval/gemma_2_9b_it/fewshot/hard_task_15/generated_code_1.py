import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    main_path = GlobalAveragePooling2D()(input_layer)
    main_path = Dense(units=32, activation='relu')(main_path)
    main_path = Dense(units=3, activation='relu')(main_path)
    main_path = Reshape(target_shape=(32, 32, 3))(main_path)

    # Branch Path
    branch_path = input_layer

    # Add outputs
    merged_path = Add()([main_path, branch_path])

    # Final Layers
    merged_path = Dense(units=64, activation='relu')(merged_path)
    output_layer = Dense(units=10, activation='softmax')(merged_path)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model