# NLP Chatbot 
 
A NLP Chatbot trained via Forward Neural Network using Pytorch. Trained over a custom chatbot_tags.json file.

## JSON File Data
The chatbot_tags.json file contains the intents on which the model is trained with each intent given a specific set of tag, patterns, and reponses. The model classifies each pattern by tokenizing the sentence and stemming out the words, splitting out a random response that correlates to each bag of words. 

## Flask or GUI Interface
The app is created with Flask, but it's not uploaded to Heroku or any cloud platform as of yet.

To run the app locally, install Flask in your server via the command:
> python -m pip install flask

Run the Flask server via the command:
> python -m flask run

If the app is running via Flask, comment out this line of code in the ```chat.py``` file for better transition.

```
bot = ChatBot()
while True:
    your_response = input("You: ")
    if your_response == "stop":
        break
    print(bot.response(your_response))
```
## Running Code
- The ```train.py``` file train the chatbot_tags.json file based on the Forward Neural Network. You can change the epochs, learning rate, etc. accordingly. The current epochs number is 1000 and the learning rate is 0.001, which gives an decent accurancy in response prediction.

- The ```model.py``` file set up the neural network pattern with three linear layers.

- The ```nltk_utils.py``` file tokenize (splitting the sentences into smaller words and phrases), stem (splitting words to be identified by their roots), and bag the words based on the frequency a certain word in a sentence.
