class Greeter(object):

    # constructor
    def __init__(self, name):
        self.name = name  # create an instance variable

    # instance method
    def greet(self, loud=False):
        if loud:
            print('Hello, %s !' % self.name.upper())
        else:
            print('Hello, %s !' % self.name)


if __name__ == '__main__':
    g = Greeter('fred')
    g.greet()
    g.greet(loud=True)
