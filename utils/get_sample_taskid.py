from utils.getallfiles import GetAllFiles

def get_sample_taskid(directory):
    
    get_all_files = GetAllFiles(directory, '.py')
    files = get_all_files.get_all_files_in_directory()

    ids = []
    for file in files:
        id = file.split('/')[-2].split('_')[-1]
        ids.append(id)
        
    return ids