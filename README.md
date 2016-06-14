# Kiedis

Pipeline for training an LSTM to generate Red Hot Chili Pepper lyrics, adapted from a similar tutorial from keras, which is the key dependency for running.  If you have keras installed (preferably with a gpu backend), just execute

		python main.py

from the command line, or preferably enter an iPython terminal and execute

		run main.py

This will generate data (if it does not already exist in the folder) by mining the lyrics from all RHCP pages on azlyrics.com, then train an RNN on this data to perform character prediction from input character sequences.

# Future Development

* Cleaning up the data (rerunning get_data.py and figuring out the BadStatusLine error it throws sometimes now.  I'm pretty sure azlyrics.com is blocking every IP address this gets run from, so should probably find a way to automatically proxy that)
	* Deal with songs that were not written by Anthony Kiedis (any one of the 50 Hendrix covers they do).  Though probably not the lowest hanging fruit (in theory, these should be representative of lyrical styles that influenced him and are somewhat similar to the ones we would write.  Could be an interesting application of hidden state clustering, see below)

* Training a model on a larger dataset to give it the basics of English, then fine-tuning on Anthony Kiedis lyrics

* Explore randomization of sequence length during text generation.  Also narrow down grid search of best temperature values, clearly somewhere between 0.5 and 1

* Hold out set to avoid overfitting?  Dealing with repetitive lines like in "Give It Away"?
	* Right now choruses are repeated, does this make sense?

* Use final hidden state from passing an entire song as a sequence for stuff like clustering, etc. to explore impacts on or applications to a "lyrics space"
	* tSNE 2D embeddings?

* generalize get_data to run for arbitrary artists on azlyrics.com

* use word embeddings instead of one-hot vectors

* feed non Anthony Kiedis lyrics as generating seed and see how he would finish, say, a Bob Dylan line
