import os
from datetime import datetime
from eccodes import *

def initialize_files(cfgstr):
    files = []
    data = []
    input_path = cfgstr['input']['path']
    print(f'Initializing files from input path: {input_path}')
    for file in os.listdir(input_path):
        if file.endswith('.bufr'):
            file_path = os.path.join(input_path, file)
            files.append(datetime.strptime(file[5:-5], "%Y%m%d%H"))
            data.append(file_path)
            print(f'Found BUFR file: {file_path}')
    return files, data

def read_bufr_messages(file):
    messages = []
    try:
        with open(file, 'rb') as f:
            nmsg = codes_count_in_file(f)
            for i in range(nmsg):
                bufr = codes_bufr_new_from_file(f)
                if bufr is None:
                    print(f'Failed to load BUFR message from {file}')
                    continue
                try:
                    codes_set(bufr, 'unpack', 1)
                    messages.append(bufr)
                except CodesInternalError as err:
                    print(f"Error decoding BUFR message from {file}: {err}")
                    codes_release(bufr)
    except Exception as e:
        print(f"Error reading BUFR messages from {file}: {e}")
    return messages

def get_all_keys(bufr):
    keys = []
    try:
        iter_id = codes_bufr_keys_iterator_new(bufr)
        while codes_bufr_keys_iterator_next(iter_id):
            key_name = codes_bufr_keys_iterator_get_name(iter_id)
            keys.append(key_name)
        codes_bufr_keys_iterator_delete(iter_id)
    except CodesInternalError as err:
        print(f"Error creating keys iterator: {err}")
    return keys

def decode_bufr_message(bufr):
    keys = get_all_keys(bufr)
    data = {}
    for key in keys:
        try:
            value = codes_get(bufr, key)
            data[key] = value
        except CodesInternalError as err:
            print(f'Error with key="{key}": {err.msg}')
            if 'Passed array is too small' in str(err):
                try:
                    value = codes_get_array(bufr, key)
                    data[key] = value
                except Exception as array_exception:
                    print(f"Error with array key={key}: {array_exception}")
    return data
