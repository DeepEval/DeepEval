import os
import json
from utils.getallfiles import GetAllFiles


class Report:
    """
    Count the number of errors in the static analysis report.
    """

    def __init__(self, save_path_name):
        self.save_path_name = save_path_name

    def count(self,type,message_total_counts):
        if type == 'convention':
            message_total_counts['convention'] += 1
        elif type == 'warning':
            message_total_counts['warning'] += 1
        elif type == 'error':
            message_total_counts['error'] += 1
        elif type == 'fatal': 
            message_total_counts['fatal'] += 1
        elif type == 'refactor':
            message_total_counts['refactoring'] += 1

        return message_total_counts

    def compute_score(self,code_lines,message_total_counts):
        """
        compute the score of the static analysis report
        ref:
        https://pylint.pycqa.org/en/latest/user_guide/configuration/all-options.html#evaluation
        """
        convention_count = message_total_counts.get('convention')
        warning_count = message_total_counts.get('warning')
        error_count = message_total_counts.get('error')
        fatal_count = message_total_counts.get('fatal')
        refactor_count = message_total_counts.get('refactoring')

        score = max(0, 0 if fatal_count else 10.0 - ((float(5 * error_count + warning_count + refactor_count + convention_count) / code_lines) * 10))

        return score

    def report(self,module_dict_list):
        global average_score, score, code_lines
        get_all_files = GetAllFiles(self.save_path_name, file_type='json')
        json_files = get_all_files.get_all_files_in_directory()

        message_counts, message_counts_socre = {}, {}
        score_list,message_counts_list = [],[]


        for json_file in json_files:

            if  os.path.getsize(json_file) != 0:
                with open(json_file, 'r') as file:
                    data_list = json.load(file)

                for module_dict in module_dict_list:
                    code_lines = module_dict['total_code_lines']
                    if code_lines != 0:
                        message_total_counts = {
                            'convention': 0, 
                            'warning': 0, 
                            'error': 0, 
                            'fatal': 0, 
                            'refactoring': 0
                        }
                        for data in data_list:
                            if module_dict['module_name'] == data.get('module'):
                                type = data.get('type')
                                message = data.get('message')
                                symbol = data.get('symbol')

                                if symbol != 'false-positive-tensorflow-import':
                                    message_counts = self.count(type,message_total_counts)
                                    score = self.compute_score(code_lines, message_counts)

                        score_list.append(score)
                        message_counts_list.append(message_counts)

                if len(score_list) == 0:
                    average_score = None
                else:
                    average_score = sum(score_list) / len(score_list)

                total_counts = {'convention': 0, 'warning': 0, 'error': 0, 'fatal': 0, 'refactoring': 0}
                for count in message_counts_list:
                    for key, value in count.items():
                        total_counts[key] += value

                total_counts['valid_pyfiles'] = len(score_list)
                total_counts['average_score'] = average_score 
                message_counts_socre.update(total_counts)

        return message_counts_socre


