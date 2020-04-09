from fracPy.engines import engine_dict

def reconstruct(obj):
    obj = engine_dict[obj.param.engine](obj)
    return obj