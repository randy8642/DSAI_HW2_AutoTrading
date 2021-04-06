import numpy as np


class stock():
    def __init__(self, hold=0, price=0) -> None:
        self.hold = hold
        self.price = price

        pass

    def setNewPrice(self, price):
        self.nextPrice = price
        return self.action()

    def action(self) -> int:
        '''
        買賣交易
        ---

        持有數量       | 今天收盤  | 預測明天收盤 | 動作 
        -------------- | :-----: | :-----: | :----:
        1    | 低 |  高 |    無 0
        1    | 高 |  低 |    賣 -1
        0    | 低 |  高 |    買 1
        0    | 高 |  低 |    賣 -1
        -1    | 低 |  高 |    買 1
        -1    | 高 |  低 |    無 0

        '''
        if self.hold == 1:
            if self.price < self.nextPrice:
                return 0
            elif self.price > self.nextPrice:
                return -1
        elif self.hold == 0:
            if self.price < self.nextPrice:
                return 1
            elif self.price > self.nextPrice:
                return -1
        elif self.hold == -1:
            if self.price < self.nextPrice:
                return 1
            elif self.price > self.nextPrice:
                return 0
