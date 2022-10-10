def expr_io(filename, expression=''):
    """
    Import or export kriging expression as .txt file.

    Parameters
    ----------
    filename: String.
        filename with extension txt.
    expression: String.
        kriging expression.
        Import txt file which contains a kriging expression if Empty string '' is taken.
        Export the kriging expression when expression is given as a variable name of
        kriging expression. The default is ''.

    Returns
    -------
    String.
        kriging expression.
    """
    if expression != "":
        with open(filename, "w") as f:
            f.write(str(expression))
            f.close
            print('The kriged function has been saved.')
    else:
        file = open(filename, "r")
        expression = file.readlines()
        return expression[0]