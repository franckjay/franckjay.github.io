---
layout: post
comments: true
title:  "Build a React App that Suggests Novel Novels with a Python Graph"
excerpt: "Recommendations can be more interesting!"
date:   2019-11-1
---

There are a lot of great resources for learning Data Science techniques out there. MOOCS, blogs, tutorials, and 
bootcamps are all great avenues for learning. Personally, I learn best by working on projects that I find interesting 
and engaging. Nothing motivates me more to push my level of understanding to new heights than by working on an intriguing 
problem. How do I identify new avenues that I can explore? Generally by reading other people's work!

This idea was first realized by reading Vicki Boykis' great 
[blog](https://vicki.substack.com/p/big-recsys-redux-recs-at-netflix) post about the changing nature of 
Netflix's recommendation engine. In short (sorry for the hack job, Vicki), gone are the days of minimizing the 
RMSE of a user's rating preferences with Matrix Factorization and Deep Learning. Recommendations are still partly an 
art form, with context, nuance, and design all playing a large part in providing a great experience to the user. 
In addition, the business value of a recommendation is not simply to provide the best content to the user. In the 
case of Netflix, it may be to provide the best _Netflix_ content to the user.

I had the great pleasure of attending [RecSys 2019](https://recsys.acm.org/recsys19/) in Copenhagen this year. 
The winner for [Best Paper](https://arxiv.org/abs/1907.06902) was one that was *not* a State of the Art Neural 
Network approach. Instead, it was a careful examination of previous, open source recommendation systems, open source
 datasets, and their actual performance in a head-to-head matchup. The results, you may ask? Simple models (Graph based, 
 User/Item Nearest Neighbors, Top Popular items) nearly always outperform the best deep neural networks.

Needless to say, I have been rethinking Recommender Systems in light of these revelations. What do I, as a user, 
want from YouTube's recommendation engine? What about a Search Engine like Ecosia/DuckDuckGo? In the former example, 
I might want to see videos that entertain/inform me in that moment. In the latter example, I want to find the most 
relevant information, quickly. In either case, the recommended items may not be the most popular, highly rated, or 
controversial that advertisers find profitable. 

So what do I truly want? How can I build a system that optimizes for this elusive goal? There was a landmark moment of 
clarity in my mind as I was walking to the library the other day: I absolutely love when I find a new author or book 
that I had not considered or heard of. The sheer discovery, novelty of that moment can inspire weeks of happiness as 
I flip through the pages.

Time spent reading a book is an investment, and so it can be a difficult proposition to try out anything new 
(explore vs. exploit). A system that just recommends random items may provide unexpected results, but they may
 not be good ones. I want to have my cake and eat it, too. I want to find great books that I would never have 
 considered before.

My idea is simple:
* People who read and like the same books as I may have roughly similar tastes to me
* These same neighbors probably have read multiple books, some of which I have not read
* Some of these unread books may be so obscure/unpopular, that I would bue unlikely to discover them on my own

None of the above statements should be that surprising to you, particularly the first two points. The last point, 
of course, is the most fundamental to this work. What if I built a Graph database of books, authors, and readers, 
and then walked along the nodes of the graph to find these hidden gems of books?

The site GoodReads has published a dataset of 10,000 books, 50,000+ users, and nearly 6 million ratings (1-5 Stars) 
of books. Surely I could find something interesting in this treasure trove! I have never built a real graph before in
 my life, so I thought this would be a great trial by fire for myself. Nodes and Edges, how hard could it be? The Nodes
  of this Graph are the `Users`, `Books`, and `Authors` of those books. 
```buildoutcfg
class User(object):
    def __init__(self,user_id):
        self.user_id = user_id
        self.shelf = [] # Books read
        self.author_list = [] # Authors read

class Book(object):
    def __init__(self, book_id, Author, ratings_5, popularity, image_url):
        self.book_id = book_id
        self.author = Author
        self.author_id = Author.author_id
        self.ratings_5 = ratings_5 # Number of people that rated the book a 5
        self.popularity = popularity # What fraction of ratings does this book have?+
        self.image_url = image_url
        self.reader_list = [] #Users that read the book

    def add_reader(self,User):
        if User not in self.reader_list:
            self.reader_list.append(User) # User read this book

class Author(object):
    def __init__(self, author_id):
            self.author_id = author_id
            self.reader_list = [] #People who read the book

    def add_reader(self,User):
        if User not in self.reader_list:
            self.reader_list.append(User) # User read this book
```

The Edges are the ratings associated with each `(User, Book, Author)` tuple:
```buildoutcfg
class Read(object):
    def __init__(self, User, Book, Author, rating=None):
        """
        The edge connecting User, Book, and Author nodes
        """
        if Book not in User.shelf:
            User.shelf.append((Book, rating)) # User read this book and rated it.
        if Author not in User.author_list:
            User.author_list.append(Author)

        self.user = User
        self.book = Book
        self.author = Author
        self.rating = rating # Optional

        Book.add_reader(User)
        Author.add_reader(User)
```

I first created 4 Classes to account for my Graph components. Each Node will also have a List associated with it:
 `Users` will have a shelf of `Books`, `Authors` will have their bibliography, and `Books` will have their 
 audience. The `Graph` will connect all of these objects together into a cohesive unit that can be traversed.
 ```buildoutcfg
class Graph(object):
    def __init__(self, reads):
        """

        """
        # Edges in our graph
        self.reads = reads


    def _find_a_user(self, input_User, debug=False):
        """
        Find a user in the graph N hops from an author in the
        user's list of read authors:
        """
        _author_list = input_User.author_list
        if debug:
            print ("Method: _find_N_user : `_author_list`: ", _author_list)
        # Do we have any authors in the list?
        _n_authors = len(_author_list)
        if _n_authors > 0:
            # Pick an Author. Any Author.
            _reader_list = None
            while _reader_list == None:
                _next_author = _author_list[np.random.randint(_n_authors)] #Inclusive, random integer
                if debug:
                    print ("Method: _find_N_user : `_next_author` : ", _next_author)
                if len(_next_author.reader_list) > 1: # Is there anyone in this bucket?
                    _reader_list = _next_author.reader_list # Who reads this Author?

            _next_User = None
            while _next_User == None:
                _choice = _reader_list[np.random.randint(len(_reader_list))]
                if _choice != input_User:
                    _next_User = _choice # Make sure we do not pick ourselves
            if debug:
                print ("Method: _find_N_user : `_next_User`: ", _next_User)
            return _next_User # We finally made a choice!
        else:
            return None

    def _book2book(self, input_Book, N=3 , debug=False):
        """ Sanity Checker:
        Developer Function to quickly get similar, unpopular books
        recommended that is not based on a User. Simply input a book,
        go up the Author tree, and then find a random user, and their
        unpopular book.

        This is quicker, in theory, for testing out the predictions, as you
        don't have to build a User + Read objects for the Graph.
        """
        def _sort_tuple(tuple_val):
            """ For sorting our unread list based on popularity
                [(book1,popularity1),...,(bookn,popularityn)]
            """
            return tuple_val[1]

        out_recs = []
        for i in range(N):
            _reader_list = input_Book.reader_list
            _len_rl = len(_reader_list)
            _rand_User = _reader_list[np.random.randint(_len_rl)]
            _list = [(book, book.popularity, rating) for book, rating in _rand_User.shelf
                    if rating > 4] #NB; No filtering on read books, as no input user.
            _list = sorted(_list, key=_sort_tuple, reverse=False)
            unpopular_book, popularity, rating = _list[0]
            out_recs.append(unpopular_book)

        return out_recs

    def _find_a_book(self, input_User, two_hop=False, debug=False):
        """
        Given a user, recommend an unpopular book:
        1) Take an input_user, go to an Author node, grab another user that has
            read that author.
        2) Grab that User's book list and compare it to the input user.
        """
        def _sort_tuple(tuple_val):
            """ For sorting our unread list based on popularity
                [(book1,popularity1),...,(bookn,popularityn)]
            """
            return tuple_val[1]

        if debug:
            print ("Method: _find_a_book : `input_User`: ", input_User)

        _next_User = self._find_a_user(input_User, debug=debug)

        if two_hop:
            try:
                _two_hop = self._find_a_user(_next_User, debug)
                _next_User = _two_hop if _two_hop != input_User else _next_User
            except Exception as e:
                if debug:
                    print ("Method: _find_a_book : Exception at `two_hop`: ", input_User, e)

        if debug:
            print ("Method: _find_a_book : `_next_User`: ", _next_User)

        counter= 0
        while counter < 100:
            counter+=1

            """
            First, let's see how many books this user has read that
            the input_User has not AND is rated above 4 stars.
            NB: We could also add a maximum popularity here, just in case!
            This will form our set from which we can find unpopular books:
            """
            try:
                _unread_list = [(book, book.popularity, rating) for book, rating in _next_User.shelf
                                if book not in [_books for _books, _rating in input_User.shelf] and rating > 4]
                _n_unread = len(_unread_list)
                if debug:
                    print ("Method: _find_a_book : Length of the unread shelf: ", _n_unread)
            except Exception as e:
                print ("Method: _find_a_book : `_unread_list` threw an exception: ", _next_User, e)

            """
            Now, we take our unsorted, unread list of books, and sort them
            in ascending order. The first entry should be our best bet!
            """
            try:
                _unread_list = sorted(_unread_list, key=_sort_tuple, reverse=False)

                if debug:
                    if _n_unread > 1:
                        print ("Method: _find_a_book : Most unpopular book title, popularity, and rating ",
                               _unread_list[0][0].book_id, _unread_list[0][1])
                        print ("Method: _find_a_book : Most popular book title and popularity ",
                               _unread_list[_n_unread-1][0].book_id, _unread_list[_n_unread-1][1])
                    else:
                        print ("Method: _find_a_book : Most unpopular book title and popularity ",
                               _unread_list[0][0].book_id, _unread_list[0][1])
            except Exception as e:
                if debug:
                    print ("Method: _find_a_book : `_unread_list` sorting threw an exception: ", e)

            # So we may have found a good, rare book. Return it!
            unpopular_book, popularity, rating = _unread_list[0]
            if unpopular_book != None:
                return unpopular_book
        # Base case: We did not find any good books.
        return None

    def GrabNBooks(self, input_User, N=3, debug=False):
        """
        Our main class to find three unpopular books. Relies on two helper classes:
            _find_a_book: Grabs a rare books from a neighbor that reads similar books to you
            _find_a_user: Finds the neighbor to a book from your collection!

        If you enable two_hop = True in your calls, it can help preserve the privacy of your users,
            as you really start to jump around the graph. Want more privacy? Enable more random jumps.
        """
        if debug:
            print ("Method: GrabThreeBooks : Beginning :", input_User.user_id)

        RareBooks = []
        counter = 0
        while counter < 100:
            """
            try:
                _book = self._find_a_book(input_User, debug)

                if _book != None:
                    RareBooks.append(_book)
            except Exception as e:
                if debug:
                    print ("Method: GrabThreeBooks : Exception = ", e)
            """
            _book = self._find_a_book(input_User, debug=debug)
            RareBooks.append(_book)
            if len(RareBooks) == N:
                return RareBooks
            # Increase the counter so that we don't get stuck in a loop
            else:
                counter+=1
        #Base case in case something goes wrong...
        return None
```

The above block of code is pretty chunky, and there are a number of methods inside that are not useful here.
We will focus on one that can be directly used with an easy, consumer facing React App: `_book2book`.
So let's say that we can now walk around our Graph, and everything compiles fine. How might we find meaningful 
`Books` to present to a `User`? In our mind, let us take a hypothetical walk:
1. You, as a `User`, loves a `Book` by `Author` (e.g., a Five star rating)
2. This `Author` has many fans, and she has written similar `Books` you may also love
    1. However, you probably are already aware of her other `Books`, so these are not Novel Novels
    2. Her fans may have read `Books` you have not, however
3. So we find a new `User` that also likes this particular `Author`
4. From that new `User`, we look at their `Book` shelf and grab only the most highly rated ones
5. This set of `Books` has a pretty good chance of being interesting to the initial user. We can:
    1. Recursively `GO TO` Step 2) and find another set of good `Books` from _another_ `User`, N times
    2. Or, just return the least popular (but still great) `Book` as our recommendation

Now we just have to return this `Book` to our `User` as a possible great discovery! We are missing the 
key ingredient to this project, though. We need the App to deliver these interesting `Book`s. Let's start with
the part that is still Python based, for familiarity: we need to build the back end of our App that handles API 
requests, and so of course we are going to use [Flask]("https://palletsprojects.com/p/flask/"). Flask allows us to 
make a small web server in Python, and is pretty easy to use. I found two tutorials to be extremely useful, and they 
were nicely linked to one another: a Flask [tutorial]("https://www.youtube.com/watch?v=Urx8Kj00zsI&t=421s"),
 and a [React App]("https://www.youtube.com/watch?v=06pWsB_hoD4&t=233s") that calls the aforementioned Flask app.
 
So we first make a new directory that we will call `api/`. Inside this folder, we will have two python files. 
`__init__.py` will build up the basics of our Flask app, import the necessary variables, etc.:
```buildoutcfg
import logging as logger
from flask import Flask

def create_app():
    app = Flask(__name__)

    from .app import main
    app.register_blueprint(main)
    
    logger.debug("App registered")
    return app
``` 

and then an `app.py`  (or whatever you want to call it):
```buildoutcfg
import logging as logger
from flask import Blueprint, jsonify, request
from .GoodReadsGraph import BuildGraph

main = Blueprint('main', __name__)
logger.debug("App starting")

# Build our Graph
BigGraph, titles_dict = BuildGraph()
logger.debug("Graph is built service")

output_URL = ""

@main.route('/input_book', methods=['POST'])
def input_book():
    # We get a request from our React App
    logger.debug("Starting service")
    _book = request.get_json()
    # We extract the title from our JSON
    _book_title = str(_book["title"])
    logger.debug("book: ", _book_title)
    # We use our dictionary to pull out a book Object
    try:
        _book_object = titles_dict[_book_title.lower()]["Book"]
        logger.debug(_book_object)
        # Grab the first entry, return it
        book_list = BigGraph._book2book(_book_object, N=1)
        book = book_list[0]
        logger.debug(book)
        global ouput_URL
        ouput_URL = str(book.image_url)
        # Return the image URL to be displayed on our app
        #    NB: POST does not support anything else.
        return "Done", 201
    except Exception as e:
        logger.warning("Encountered exception: ", e)
        return "Error", 400

@main.route('/novel_novel', methods=['GET'])
def novel_novel():
    logger.debug("GET method returning output_url")
    return jsonify({"image_url": ouput_URL}), 200
``` 

Inside the `app.py`, we are doing a number of important things. First, we call our python code that builds us our 
graph database for finding new books. It also sets up the `Blueprint` of our backend, so when we make API calls to this
service, we are pointing in the correct direction. Let's unpack these things at a very basic level: 
* Our app name is initialized to be `@main`, but we could have other stuff here as well
* We are hosting two `.routes` in our `@main` app:  `/input_book` and `/novel_novel`
* You can think of these as separate pages on our Web app, and they both handle different behaviors
* The HTTP methods of `GET` and `POST`, which get something from our backend or post something to it (good terminology!)

For the `/input_book` call, what we are doing is receiving a raw text input from the
 `User` on the webpage, packaging it as a JSON that looks like `{"book": "book_title"}`, and then updating the 
 back end server. In this case, we are using this initial book to find a similar, rare book the `User` may enjoy.
 We call out Graph with this input, and set a `global` python variable to the image URL of the output book.
 
Now you can imagine that the WebApp would make a 2nd API call to the Flask App, saying "What is the output of that 
input?". This is our `GET` call in `/novel_novel`, which returns a JSON as well.

 

