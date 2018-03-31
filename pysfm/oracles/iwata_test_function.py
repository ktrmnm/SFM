class IwataTestFunction:
    def __init__(self, n=5):
        self.n = n

    @property
    def name(self):
        return "iwata_test_function"

    @property
    def data(self):
        return { 'n': self.n }
