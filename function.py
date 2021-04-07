
class stock():
    def __init__(self) -> None:
        self.hold = 0

        self.actions = []
        pass


    def trade(self, predict) -> int:
        '''
        買賣交易
        ---
        input\\

        -1:跌
        0:持平
        1:漲

        option\\

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
            if predict == 1:
                self.actions.append(0)
            elif predict == 0:
                self.actions.append(0)
            elif predict == -1:
                self.actions.append(-1)
        elif self.hold == 0:
            if predict == 1:
                self.actions.append(1)
            elif predict == 0:
                self.actions.append(0)
            elif predict == -1:
                self.actions.append(-1)
        elif self.hold == -1:
            if predict == 1:
                self.actions.append(1)
            elif predict == 0:
                self.actions.append(0)
            elif predict == -1:
                self.actions.append(0)

        # 
        self.hold += self.actions[-1]

        return self.actions[-1]

# USAGE EXPAMPLE
A = stock()

for i in [0,1,1,-1,-1,-1,-1,-1]:
    print('---------------')
    A.trade(i)
    print(A.hold)
    print(A.actions)