def get_layer_list(model):
    """
    returns a list of all layers in the input model, including layers in any nested models therein
    layers are listed in forward-pass order
    """
    arr = []
    final_arr = []
    try:
        arr = [m for m in model.modules()]
        if len(arr) > 1:
            for module in arr[1:]:
                final_arr += get_module_list(module)
            return final_arr
        else:
            return arr
    except:
        raise(NotImplementedError("expected module list to be populated, but no '_modules' field was found"))

def replace_layer(old_layer, new_layer, model):
    """
    search through model until old_layer is found, and replace it with new layer;
    returns True is old_layer was found; False otherwise
    """
    try:
        modules = model._modules
        for k in modules.keys():
            if modules[k] == old_layer:
                modules[k] = new_layer
                return True
            else:
                if replace_layer(old_layer, new_layer, modules[k]):
                    return True
        return False
    except:
        raise(NotImplementedError("expected module list to be populated, but no '_modules' field was found"))

def is_same_type(layer1, layer2):
    """
    returns True if layer1 and layer2 are of the same type; false otherwise
    if a class is input as layer2 [e.g., is_same_type(my_conv_layer, Conv2d) ], the type defined by the class is used
    if a string is input as layer2, the string is matched to the name of the class of layer1
    """
    if type(layer2) == type:
        return type(layer1) == layer2
    elif type(layer2) == str:
        return type(layer1).__name__ == layer2
    else:
        return type(layer1) == type(layer2)

def same_type_fn(layer1):
    """
    curried version of is_same_type; returns a function f such than f(layer2) <-> is_same_type(layer1, layer2)
    """   
    return lambda x: is_same_type(x, layer1)
        

def replace_layers_on_condition(model, condition_fn, new_layer_factory_fn):
    """
    search through model finding all layers L such that conditional_fn(L), and replace them with new_layer_factory_fn(L)
    returns true if at least one layer was replaced; false otherwise
    """
    has_replaced = False
    try:
        modules = model._modules
        for k in modules.keys():
            if condition_fn(modules[k]):
                modules[k] = new_layer_factory_fn(modules[k])
                has_replaced = True
            else:
                has_replaced = replace_layers_on_condition(modules[k], condition_fn, new_layer_factory_fn) or has_replaced
        return has_replaced
    except:
        raise(NotImplementedError("expected module list to be populated, but no '_modules' field was found"))

def replace_layers_on_conditions(model, condition_factory_pairs):
    """
    search through model finding all layers L such that for some ordered pair [conditional_fn, new_layer_factory_fn] in condition_factory_pairs,
    conditional_fn(L), and replace them with new_layer_factory_fn(L)
    layers will only be replaced once, so the first conditional for which a layer returns true will be last conditional to which it is compared
    returns true if at least one layer was replaced; false otherwise
    """
    has_replaced = False
    try:
        modules = model._modules
        for k in modules.keys():
            for (condition_fn, new_layer_factory_fn) in condition_factory_pairs:
                if condition_fn(modules[k]):
                    modules[k] = new_layer_factory_fn(modules[k])
                    has_replaced = True
                    break
            else:
                has_replaced = replace_layers_on_conditions(modules[k], condition_factory_pairs) or has_replaced
        return has_replaced
    except:
        raise(NotImplementedError("expected module list to be populated, but no '_modules' field was found"))

def replace_layers_of_type(model, layer_type, new_layer_factory_fn):
    """
    search through model finding all layers L of type layer_type and replace with new_layer_factory_fn(L)
    returns true if at least one layer was replaced; false otherwise
    """
    return replace_layers_on_condition(model, lambda x: is_same_type(x,layer_type), new_layer_factory_fn)

def replace_layers_of_types(model, type_factory_pairs):
    """
    search through model finding all layers L such that for some ordered pair [layer_type, new_layer_factory_fn] in type_factory_pairs, 
    L is of type layer_type, and replace with new_layer_factory_fn(L)
    returns true if at least one layer was replaced; false otherwise
    """
    condition_factory_pairs = [(same_type_fn(layer_type), new_layer_factory_fn) for (layer_type, new_layer_factory_fn) in type_factory_pairs]
    return replace_layers_on_conditions(model, condition_factory_pairs)