get_data.py mine's the Red Hot Chili Peppers'page on azlyrics.com and rips the text from every song that doesn't toss it an error.  Probably much better ways to handle this.

main.py rips off a keras example for Nietzsche text generation, and tends to perform much worse with worse overfitting.  main2.py is the updated version that does a lot more to randomize and speed up the training process to perform more robustly and avoid overfitting.

# Future Development

-Cleaning up the data (rerunning get_data.py and figuring out the BadStatusLine error it throws sometimes now.  It's possible that azlyrics.com shuts out ip addresses it thinks are running in an automated fashion)

-Training a model on a larger dataset to give it the basics of English, then fine-tuning on Anthony Kiedis lyrics

-Explore randomization of sequence length during text generation.  Also narrow down grid search of best temperature values, clearly somewhere between 0.5 and 1

-Use final hidden state from passing an entire song as a sequence for stuff like clustering, etc. to explore impacts on or applications to a "lyrics space"

		-tSNE 2D embeddings?

-generalize get_data to run for arbitrary artists on azlyrics.com

-use word embeddings instead of one-hot vectors

-feed non Anthony Kiedis lyrics as generating seed and see how he would finish, say, a Bob Dylan line
