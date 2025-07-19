import inspect


def make_params_strings(func):
    func_params = inspect.signature(func).parameters
    func_params_str = ', '.join(
        [k if v.default == inspect._empty else f'{k}={v.default}' for k, v in func_params.items()]
    )
    func_names_params_str = ', '.join(func_params.keys())
    return func_params_str, func_names_params_str


if __name__ == "__main__":
    def aux(x, y, z=1):
        pass
    print(make_params_strings(aux))

