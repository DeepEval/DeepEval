import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Activation, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Concatenate, BatchNormalization, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(x):
        path1 = GlobalAveragePooling2D()(x)
        path1 = Dense(128, activation='relu')(path1)
        path1 = Dense(64, activation='relu')(path1)
        
        path2 = GlobalMaxPooling2D()(x)
        path2 = Dense(128, activation='relu')(path2)
        path2 = Dense(64, activation='relu')(path2)
        
        merged = Add()([path1, path2])
        attention_weights = Activation('sigmoid')(merged)
        output = Multiply()([x, attention_weights])
        return output
    
    output = block1(input_layer)

    # Block 2
    avg_pool = AveragePooling2D(pool_size=(7, 7), strides=1)(output)
    max_pool = MaxPooling2D(pool_size=(7, 7), strides=1)(output)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    conv = Conv2D(filters=32, kernel_size=(1, 1), activation='sigmoid')(concat)
    normalized_features = Multiply()([output, conv])

    # Final branch
    final_branch = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(normalized_features)
    final_output = Add()([normalized_features, final_branch])
    final_output = Activation('relu')(final_output)

    # Flatten and fully connected layers
    flatten = Flatten()(final_output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model