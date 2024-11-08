class Follicle:
    def __init__(self, size):
        self.size = size

    def increase(self):
        self.size = self.size + 1

    def __str__(self):
        return str(self.size)
