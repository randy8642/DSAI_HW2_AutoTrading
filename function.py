

class stock():
    def __init__(self, hold=0, init_price=0) -> None:
        self.hold = hold
        self.price = init_price

        self._actions = []
        

    def setNewPrice(self, price):
        self.nextPrice = price

    def trade(self) -> int:
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
                self._actions.append(0)
            elif self.price > self.nextPrice:
                self._actions.append(-1)
        elif self.hold == 0:
            if self.price < self.nextPrice:
                self._actions.append(1)
            elif self.price > self.nextPrice:
                self._actions.append(-1)
        elif self.hold == -1:
            if self.price < self.nextPrice:
                self._actions.append(1)
            elif self.price > self.nextPrice:
                self._actions.append(0)

        return self._actions[-1]

    def getActions(self):
        return self._actions
