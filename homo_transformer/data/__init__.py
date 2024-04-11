from . import apolloscape_dataset


MODULES = {
    'apolloscape': apolloscape_dataset,
}


def get_dataset_module_by_name(name):
    return MODULES[name]
