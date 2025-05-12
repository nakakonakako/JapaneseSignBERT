import json
import gc
import torch

def my_import(name):
    """
    動的にモジュールまたはオブジェクトをインポートする

    モジュールのフルネームまたはモジュール内のオブジェクトのフルネーム（文字列として）が与えられた場合、
    この関数はそれをインポートしてモジュールまたはオブジェクトを返します。
    
    Parameters:
    name (str): インポートするモジュールまたはオブジェクトのフルネーム、例：'numpy.array'。
    
    Returns:
    module/object: インポートされたモジュールまたはオブジェクト。
    """
    # ドットでフルネームを分割して、モジュール名とオブジェクト名を分離する
    components = name.split('.')
    # __import__関数は、モジュール名を引数に取り、そのモジュールを返す
    mod = __import__(components[0])
    # モジュールの階層をトラバースして、目的のオブジェクトに到達する
    for comp in components[1:]:
        mod = getattr(mod, comp)

    return mod

def read_json(fpath):
    with open(fpath, 'r') as fid:
        data = json.load(fid)
    return data

def read_txt_as_list(fpath):
    with open(fpath, 'r') as fid:
        data = fid.read().split('\n')
    return data

def dict_to_json_file(dict, fpath):
    with open(fpath, 'w') as fid:
        json.dump(dict, fid)


def _num_active_cuda_tensors():
    """
    Returns all tensors initialized on cuda devices
    """
    count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == "cuda":
                count += 1
        except:
            pass
    return count