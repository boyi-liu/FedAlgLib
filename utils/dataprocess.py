class Processor:
    def __init__(self):
        self.data = []

    def append(self, item, times=1):
        for _ in range(times):
            self.data.append(item)
        return

    def avg(self, p=0):
        return sum(self.data[p:]) / len(self.data[p:])

    def min(self, p=0):
        return min(self.data[p:])

    def max(self, p=0):
        return max(self.data[p:])

    def last(self):
        try:
            return self.data[-1]
        except:
            return None

    def clear(self):
        self.data = []
