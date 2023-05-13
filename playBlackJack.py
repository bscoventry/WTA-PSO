from BlackJackClass import Dealer,Player
import numpy as np
#cards = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
cardValues = [2,3,4,5,6,7,8,9,10,10,10,10,11]
cardValLen = len(cardValues)
firstCard1 = cardValues[np.random.choice(cardValLen)]
firstCard2 = cardValues[np.random.choice(cardValLen)]
ThePlayer = Player(firstCard1)
TheDealer = Dealer(firstCard2)
gameOverFlag = False
#The Game
while ~gameOverFlag:
    if Player.isOver21 and ~Dealer.isOver21:
        break
        print('YOU LOOSE SUCKER')
    if ~Player.isOver21 and Dealer.isOver21:
        break
        print('Ahhh you win Bojanglues')
    if Player.isOver21 and Dealer.isOver21:
        break
        print('Everyone Sucks Here')
    
