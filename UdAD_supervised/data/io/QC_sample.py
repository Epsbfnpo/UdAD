class QCSample:
    def __init__(self, index, b0, dwis, affine):
        self.index = index
        self.b0 = b0
        self.dwis = dwis
        self.affine = affine

        self.shape = b0.shape
