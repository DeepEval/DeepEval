class FewshotPromptDesigner:
    """
    FewshotPromptDesigner: a class to design few-shot prompts.
    Prompt:prefix-<example(requirement,code),new requirement> --> code
    """
    def __init__(self):
        pass

    def prompt(self, task_requirement):

        backgroud = """
    As a developer specializing in deep learning, you are expected to complete the code 
using Functional APIs of Keras,ensuring it meets the requirement of a task.You could draw 
inspiration from the following examples.
    """

        example_task_1 = """
    The requirement of the first example task is as follows: 
    "Please design a deep learning model for image classification using the MNIST dataset.
The model has two paths: the main path includes two 3x3 convolutional layers followed by 
max-pooling, while the branch path consists of a 5x5 convolutional layer with pooling.The 
outputs from both paths are then merged using an addition operation and then passed through a fully connected layer for classification."
    """

        example_code_1 = """
    The completed code for the first example task is as follows:
    ```python
    import keras
    from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

    def dl_model():

        input_layer = Input(shape=(28, 28, 1))
        conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_1)
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
        main_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_2)

        conv2_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
        branch_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2_1)
        adding_layer = Add()([main_path, branch_path])

        flatten_layer = Flatten()(adding_layer)
        output_layer = Dense(units=10, activation='softmax')(flatten_layer)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model
    ```
    """

        example_task_2 = """
    The requirement of the second example task is as follows: 
    "Please create a deep learning model for image classification using the MNIST dataset. The model should include
a convolutional layer followed by a pooling layer, both connected in series. Subsequently, implement a specific
block featuring four parallel paths: a 1x1 convolution, a 3x3 convolution, a 5x5 convolution, and a 1x1 max pooling
layer. Concatenate the outputs of these paths. Then, apply batch normalization and flatten the result. Finally,
the output should pass through three fully connected layers to produce the final classification."
    """

        example_code_2 = """
    The completed code for the second example task is as follows:
    ```python
    import keras
    from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

    def dl_model():     

        input_layer = Input(shape=(28, 28, 1))
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

        def block(input_tensor):

            path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
            path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
            path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
            path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
            output_tensor = Concatenate()([path1, path2, path3, path4])

            return output_tensor
        
        block_output = block(input_tensor=max_pooling)
        bath_norm = BatchNormalization()(block_output)
        flatten_layer = Flatten()(bath_norm)
        dense1 = Dense(units=128, activation='relu')(flatten_layer)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model
    ```
    """
        example_task_3 = """
    The requirement of the third example task is as follows: 
    "Please help me design a deep learning model for image classification using the MNIST dataset. The model
consists of two blocks. The first block processes its input using three average pooling layers with different
scales (pooling windows and strides of 1x1, 2x2, and 4x4). Each pooling result is flattened into a one-dimensional
vector and then concatenated. The second block splits its input into four groups, each processed using depthwise
separable convolutions with different kernel sizes. The outputs of these groups are then concatenated. Between 
these blocks, a fully connected layer and reshaping operation convert the output of the first block into a 4-dimensional
tensor suitable for the second block. Finally,the classification result is produced through a flattening layer 
followed by a fully connected layer."
    """

        example_code_3 = """
    The completed code for the third example task is as follows:
    ```python
    import keras
    import tensorflow as tf
    from keras.layers import Input, AveragePooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

    def dl_model():

        input_layer = Input(shape=(28,28,1))

        def block_1(input_tensor):
            maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
            flatten1 = Flatten()(maxpool1)
            maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
            flatten2 = Flatten()(maxpool2)
            maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
            flatten3 = Flatten()(maxpool3)
            output_tensor = Concatenate()([flatten1, flatten2, flatten3])
            return output_tensor

        def block_2(input_tensor):
            inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)
            conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
            conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
            conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
            conv4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(inputs_groups[3])
            output_tensor = Concatenate()([conv1, conv2, conv3, conv4])
            return output_tensor

        block1_output = block_1(input_tensor=input_layer)
        dense = Dense(units=64, activation='relu')(block1_output)
        reshaped = Reshape(target_shape=(4, 4, 4))(dense)
        block2_output = block_2(input_tensor=reshaped)

        flatten = Flatten()(block2_output)
        output_layer = Dense(units=10, activation='softmax')(flatten)
        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model

    ```
    """

        new_task_requirement = (
            """
    Please refer to above examples and complete a new task detailed as follows:\n"""
            + "    "
            + f'"{task_requirement}"'
        )

        code_format = """
    Please import Keras and all necessary packages, then complete python code in the 'dl_model()' function and return the constructed 'model'.
    ```python
    def dl_model():
        
        return model
    ```
    """
        prompt = (
            backgroud
            + example_task_1
            + example_code_1
            + example_task_2
            + example_code_2
            + example_task_3
            + example_code_3
            + new_task_requirement
            + code_format
        )

        return prompt

