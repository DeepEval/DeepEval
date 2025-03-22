import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # First pathway
    conv_pathway = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_pathway = BatchNormalization()(conv_pathway)
    for _ in range(3):
        conv_pathway = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_pathway)
        conv_pathway = BatchNormalization()(conv_pathway)

    # Second pathway
    max_pooling_pathway = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    max_pooling_pathway = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling_pathway)
    max_pooling_pathway = BatchNormalization()(max_pooling_pathway)
    for _ in range(3):
        max_pooling_pathway = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling_pathway)
        max_pooling_pathway = BatchNormalization()(max_pooling_pathway)

    pathway_concat = Concatenate()([conv_pathway, max_pooling_pathway])

    pathway_concat = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pathway_concat)
    pathway_concat = BatchNormalization()(pathway_concat)
    pathway_concat = Flatten()(pathway_concat)
    
    dense1 = Dense(units=128, activation='relu')(pathway_concat)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model