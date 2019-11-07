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
  of this Graph are the `Users`, `Books`, and `Authors` of those books. The Edges are the ratings associated with 
  each `(User, Book, Author)` tuple.

I first created 4 Classes to account for my Graph components. Each Node will also have a List associated with it:
 `Users` will have a shelf of `Books`, `Authors` will have their bibliography, and `Books` will have their 
 audience. The `Graph` will connect all of these objects together into a cohesive unit that can be traversed.

So let's say that we can now walk around our Graph, and everything compiles fine. How might we find meaningful 
`Books` to present to a `User`? In our mind, let us take a hypothetical walk:
1) You as a `User` loves a `Book` by `Author` (e.g., a Five star rating)
2) This `Author` has many fans, and she has written similar `Books` you may also love
    a) However, you probably are already aware of her other `Books`, so these are not Novel Novels
    b) Her fans may have read `Books` you have not, however
3) So we find a new `User` that also likes this particular `Author`
4) From that new `User`, we look at their `Book` shelf and grab only the most highly rated ones
5) This set of `Books` has a pretty good chance of being interesting to the initial user. We can:
    a) Recursively `GO TO` Step 2) and find another set of good `Books` from _another_ `User`, N times
    b) Or, return the least popular (but still great) `Book` as our recommendation
    


