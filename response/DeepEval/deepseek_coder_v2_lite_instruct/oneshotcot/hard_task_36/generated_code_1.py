import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dropout, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    main_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2))(main_path)
    main_path = Dropout(0.5)(main_path)

    # Branch pathway
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion of both pathways
    fusion = Add()([main_path, branch_path])

    # Final processing
    fusion = GlobalAveragePooling2D()(fusion)
    flatten_layer = Flatten()(fusion)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model