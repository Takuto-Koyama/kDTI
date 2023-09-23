


def convert_to_pascal_case(name):
    return ''.join(word.capitalize() for word in name.split('_'))

def create_class(name):
    class_name = convert_to_pascal_case(name)
    NewClass = type(class_name, (object,), {})
    return NewClass


