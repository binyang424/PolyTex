import re
import sympy


class save_krig(dict):
    """
    This saves a dictonary of sympy expressions to a file
    in human readable form.

    Note:
    --------
    This class is taken from: https://github.com/sympy/sympy/issues/7974.
    A bug in exec() is fixed and some modifications are made to make it
    fit for the purpose of this project (store the kriging expression).

    Example:
    --------
    >>> import sympy
    >>> a, b = sympy.symbols('a, b')
    >>> d = save_krig({'a':a, 'b':b})
    >>> d.save('name.expr')
    >>> del d
    >>> d2 = save_krig.load('name.expr')
    """

    def __init__(self, *args, **kwargs):
        super(save_krig, self).__init__(*args, **kwargs)

    def __repr__(self):
        d = dict(self)
        for key in d.keys():
            d[key] = sympy.srepr(d[key])
        # regex is just used here to insert a new line after
        # each dict key, value pair to make it more readable
        return re.sub('(: \"[^"]*\",)', r'\1\n', d.__repr__())

    def save(self, file):
        with open(file, 'w') as savefile:
            savefile.write(self.__repr__())

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'r') as loadfile:
            # Note that the variable name temp should not be the same as the other
            # local variables in the function, otherwise exec will not work and will
            # raise an NameError: name 'temp' is not defined.
            exec("temp =" + loadfile.read())
        # convert the strings back to sympy expressions and return a new save_krig.
        # This is done by calling the save_krig constructor with the new dict.
        # locals() is used to get the sympy symbols from the exec statement above.
        d = locals()['temp']
        for key in d.keys():
            d[key] = sympy.sympify(d[key])
        return cls(d)


if __name__ == '__main__':
    a, b = sympy.symbols('a, b')
    d = save_krig({'a':a, 'b':b})
    d.save('name.expr')
    del d
    d2 = save_krig.load('./name.expr')