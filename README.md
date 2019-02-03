# Project: Twitter

## 44-520 Web Mining

### Background

Twitter is a platform where a vast variety of people can publicly post content.  Just as in real life, individuals communicate differently and can be differentiated by the differences in the works they create.  For example, works authored by Charles Dickens will read very differently than those by J.K. Rowling; word choice, grammatical style, and sentence structure will be distinct.

### Your Task

Find three twitter accounts you are interested in and analyze the tweets (both content and metadata) to see if you can distinguish between authors.  You may use any combination of information about the tweets EXCEPT the twitter handle/direct author information (that's cheating!).  However, you may want to use some combination of the following:

* Lexical Diversity
* Commonly used words
* Punctuation/capitalization usage
* Tweet Length
* Device/client used (Android, iOS, Web Client)
* Sentiment Analysis (check out the course notes for this!)
* ??? Other things? Average length of word?  

You are to present your results in a Jupyter Notebook in this repository.  Your notebook must accomplish the following:

* Read your corpus of tweets
* Analyze your data based on your chosen criteria
* Display/plot your data using `matplotlib` or other python mechanism (note: this should be done live so that if your data changes so does the visualization)
* Report any conclusions you can draw from your research

When you submit your project you must read your tweets from a `pickle` file as discussed in class so the grader does not need to hit the Twitter API.
Remembetr that unless you sign up for your own twitter dev account you are sharing an API key (including the rate limiting aspect) with other members of the class; as such you will want to get your data early so you are not trying to get tweets at the last minute with everyone else.
