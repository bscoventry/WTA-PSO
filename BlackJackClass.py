class Dealer():
    
    def __init__(self,cardlist,isOver21=False,cardsPlaySum = 0,Stop = False):
        self.myCardList = cardlist
        self.cardLimit = isOver21
        self.cardSum = cardsPlaySum
        self.isStopped = Stop
    def countCards(self):
        newSum = sum(self.myCardList)
        self.cardSum = newSum
        if self.cardSum > 21:
            self.isOver21=True
        if self.cardSum > 17:
            self.stopRound()
    def addCard(self,newCard):
        self.myCardList.append(newCard)
    def stopRound(self):
        self.isStopped = True
 
class Player():
    
    def __init__(self,cardlist,isOver21=False,cardsPlaySum = 0,Stop = False):
        self.myCardList = cardlist
        self.cardLimit = isOver21
        self.cardSum = cardsPlaySum
        self.isStopped = Stop
    def countCards(self):
        newSum = sum(self.myCardList)
        self.cardSum = newSum
        if self.cardSum > 21:
            self.isOver21=True
    def addCard(self,newCard):
        self.myCardList.append(newCard)
    
