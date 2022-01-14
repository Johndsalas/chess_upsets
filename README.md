# Chess Upsets

# Introduction

I’ve always had a certain fascination with the game of chess. As an avid boardgame player I’ve played a lot of games over the years and what I believe makes chess stand out against all of them is that the two sides in chess are nearly identical. Both have two knights, eight pawns, one king etc. All begin in the same position and so on. In other games players all hold special advantages not possessed by other players that they must maximize in order to win. Some rely on dice or other change driven mechanics. While these kinds of games are fun and can have a high level of strategy, I feel that nothing emphasizes the idea of two players engaged in a one-one-one battle of wits as chess does. 
The game is extremely balanced and uses no elements of randomness (Save deciding who goes first). As such, players must rely only on their own knowledge and skill to give them advantage during the game. Because Chess is such a skill focused game, it seemed to me a natural conclusion that a more skilled player will defeat a less skilled player a large percent of the time. Large relative to any other sort of game. After all an athlete could by chance miss a shot or trip unexpectantly, luck can favor an opponent in any game in which it has a hand, but these factors would not have any influence over the chess board. I then came upon the Idea of investigating my hypotheses using a data set containing chess games that I found on Kaggle.

# Goal

Discover drivers of upsets in chess games and use those drivers to build a model to predict the whether a given game would end in upset. An upset will be defined as a lower ranked player winning a game against a higher ranked player. I will be undertaking this project in order to practice going through the data science process and to gain a better understanding of why a more skilled player might lose to a less skilled player.

# Initial Hypotheses About Drivers

* There will be few instances of upsets, possibly leading to an imbalanced data set
* As ratings for both players increase, the likelihood of an upset will decrease 
* As the margin between player ratings increase the likelihood of an upsets will decrease
* Shorter time increments will increase the likelihood of an upset
* Unranked games will have a higher likelihood of an upset than ranked games
* Games where the lower rated player has the first move will have an increased likelihood of an upset
* Some openings will be more or less prone to upsets

### Initial Thoughts

<br>

* Going into this project I am of two minds.

<br>

**First**
* Chess is a skill based game with no random elements (except assigning first move). 
* Because of this the player with the highest level of skill will win any game not determined by variation in player performance. 
* Because of this a given game will be won by the player with the highest level of skill a large majority of the time.
* If this is true conditions under which variation in performance is the highest should result in the highest likelihood of an upset.

<br>

**Second**
* It may also be the case that more skilled players are able to maintain consistency better than less skilled players under conditions that would increase variation in their performance.
* If this is true, those conditions may make upsets less likely as the variance would have a greater effect on the less skilled player than on the more skilled player.

<br>

**Moving Forward**
* Though these two schools of thought may point at differing conclusions, both seem grounded in reason and I am eager to see what the data can tell us

# The Plan

* Aquire data from Kaggle
* Prepare data
* Explore data in search of drivers of upsets and answers to the above hypotheses
*  Use drivers identified in explore to build different models
* Select the best model
* Draw conclusions

# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|Rated| True or False, The game's result is reflected in each player's rating|
|Winning Pieces| The color of pieces the winning player was moving|
|White Rating| Rating of the player moving the white pieces using the Glicko-2 rating method for games played on Lichess|
|Black Rating| Rating of the player moving the white pieces using the Glicko-2 rating method for games played on Lichess|
|Rating Difference| The difference in rating between the players in the game|
|Game Rating| The average rating of the two players in the game|
|Lower Rated White| True or False, The lower rated player is moving the white pieces|
|Opening Name| The name of the opening played in the game|
|Time Control Group| The amount of time allotted to each player to make their moves, **Standard** (60 min or more), **Rapid** (30 - 15 min), **Blitz** (5 - 3 min), or **Bullet** (2 or less), **Other** (any other time limit)|
|Upset (Target)| True or False, The lower rated player won the game|
|Additional Features|Encoded values for 'Time Control Group' and 'Opening Name'|

# Steps to Reproduce 
1) Clone this repo.
2) Acquire the data from Kaggle at https://www.kaggle.com/datasnaek/chess.
3) Put the data in the file containing the cloned repo.
4) Run notebook.
