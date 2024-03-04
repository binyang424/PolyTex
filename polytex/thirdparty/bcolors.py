class bcolors:
    """
    [Simple Python class to create colored messages for command line printing]
    (https://gist.github.com/tuvokki/14deb97bef6df9bc6553)

    Helper class to print colored output

    To use code like this, you can do something like

    print bcolors.WARNING
          + "Warning: No active frommets remain. Continue?"
          + bcolors.ENDC

    you can also use the convenience method bcolors.colored like this

    >>> print(bcolors.colored("This frumble is underlined", bcolors.UNDERLINE))

    or use one of the following convenience methods:
      warning, fail, ok, okblue, header

    Examples
    --------
    >>> print(bcolors.warning("This is dangerous"))

    Method calls can be nested too, print an underlined header do this:

    >>> print(bcolors.header(bcolors.colored("The line under this text is purple too ... ", bcolors.UNDERLINE)))
    
    """

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    # Method that returns a message with the desired color
    # usage:
    #    print(bcolor.colored("My colored message", bcolor.OKBLUE))
    @staticmethod
    def colored(message, color):
      return color + message + bcolors.ENDC

    # Method that returns a yellow warning
    # usage:
    #   print(bcolors.warning("What you are about to do is potentially dangerous. Continue?"))
    @staticmethod
    def warning(message):
      return bcolors.WARNING + message + bcolors.ENDC

    # Method that returns a red fail
    # usage:
    #   print(bcolors.fail("What you did just failed massively. Bummer"))
    #   or:
    #   sys.exit(bcolors.fail("Not a valid date"))
    @staticmethod
    def fail(message):
      return bcolors.FAIL + message + bcolors.ENDC

    # Method that returns a green ok
    # usage:
    #   print(bcolors.ok("What you did just ok-ed massively. Yay!"))
    @staticmethod
    def ok(message):
      return bcolors.OKGREEN + message + bcolors.ENDC

    # Method that returns a blue ok
    # usage:
    #   print(bcolors.okblue("What you did just ok-ed into the blue. Wow!"))
    @staticmethod
    def okblue(message):
      return bcolors.OKBLUE + message + bcolors.ENDC

    # Method that returns a header in some purple-ish color
    # usage:
    #   print(bcolors.header("This is great"))
    @staticmethod
    def header(message):
      return bcolors.HEADER + message + bcolors.ENDC
