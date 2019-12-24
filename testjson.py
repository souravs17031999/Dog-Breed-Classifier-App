import json
def class_name(index):
    with open("classes.json", "r") as read_file:
        data = json.load(read_file)
        return data[index]['class_name']

        
