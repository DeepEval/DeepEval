import re
import json


def read_humaneval(file_name):
    tasks = []
    with open(file_name, "r") as file:
        buffer = ""
        for line in file:
            buffer += line.strip()
            data = json.loads(buffer)
            prompt = data["prompt"].replace("\n", " ").strip()
            prompt = re.sub(r'\s+', ' ', prompt) 
            task = {"id": data["task_id"], "requirement": prompt}

            buffer = "" 

            tasks.append(task)

    return tasks


def read_mleval(file_name):
    tasks = []
    num = 0
    with open(file_name, "r") as file:
        buffer = ""
        for line in file:
            buffer += line.strip()
            try:
                data = json.loads(buffer)
                data["task_id"] = num 
                buffer = ""  
                task = {"id": data["task_id"], "requirement": data["intent"]}
                tasks.append(task)
                num += 1

            except json.JSONDecodeError:
                continue 
    return tasks
