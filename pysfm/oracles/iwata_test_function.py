class IwataTestFunction:
    def __init__(self, n=5):
        self.n = n

    @property
    def name(self):
        return "iwata_test_function"

    @property
    def data(self):
        return { 'n': self.n }


class GroupwiseIwataTestFunction:
    def __init__(self, n=5, k=1):
        self.n = n
        self.k = k

    @property
    def name(self):
        return "groupwise_iwata_test_function"

    @property
    def data(self):
        return { 'n': self.n, 'k': self.k }
