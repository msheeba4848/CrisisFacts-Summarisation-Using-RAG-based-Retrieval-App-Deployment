import os

def get_env_variable(var_name):
    return os.getenv(var_name, "default_value")
