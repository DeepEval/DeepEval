from itertools import product


class MergeFlattenAPISequences:

    def __init__(self, main_api_sequences_dict, sub_api_sequences_dict_lists):
        self.main_api_sequences_dict = main_api_sequences_dict
        self.sub_api_sequences_dict_lists = sub_api_sequences_dict_lists

    def merge_api_sequences(self):
        """
        Replace the custom function elements in the API sequence of the main function with the API sequence of the sub-function according to the Function_name of the sub-function, and generate all possible sequence combinations
        """
        main_api_sequences = self.main_api_sequences_dict['API Sequences']
        sub_function_map = {sub_function['Function_name']: sub_function['API Sequences'] for sub_function in
                            self.sub_api_sequences_dict_lists}

        merge_lists = []
        for main_sequence in main_api_sequences:

            sub_sequence_indices = [i for i, api in enumerate(main_sequence) if api in sub_function_map]
            sub_sequences = [sub_function_map[main_sequence[i]] for i in sub_sequence_indices]

            combined_sub_sequences = product(*sub_sequences)
            for combined_sub_sequence in combined_sub_sequences:
                target_sequence = main_sequence.copy()
                for index, combined_sub_seq in zip(sub_sequence_indices, combined_sub_sequence):
                    target_sequence[index] = combined_sub_seq
                merge_lists.append(target_sequence)

        return merge_lists

    def flatten_list(self, lists):
        flatten_list = []
        for sublist in lists:
            flat_sublist = []
            for item in sublist:
                if isinstance(item, str):
                    flat_sublist.append(item)
                else:
                    for subitem in item:
                        flat_sublist.append(subitem)
            flatten_list.append(flat_sublist)

        return flatten_list
