what we want to do for the project:
https://www.reddit.com/r/SunoAI/comments/1po4jlu/54_time/
Motivated by the online community and my personal observation using Suno, we noticed the model is more likely to succeed when told “make a waltz” than when told “make a song in ¾ time signature”, even though these describe the exact same rhythm. Also, complex time signatures like 5/4 are hard to generate with output always collapsing to 4/4, a dominant time signature in western pop music. We plan to investigate two fairness issues through a case study of text-to-music generation with different time signatures with different prompts: 1. user-experience bias: if model handles colloquial and culturally common descriptions better than formal musical terms, potentially disadvantaging musically trained users). 2. cultural bias (if models collapse time signatures rare in Western pop music to 4/4, musical traditions that rely on non-4/4 rhythms are underrepresented). We plan to present our findings as a blog post or a collab notebook. 


experiment design:

compose prompts in two different styles:

1. culturally common description
2. formal musical terms

and the each pair of two type prompts should inherently ask the text-to-music model to generate songs in the same time signature 


after the experement, here is how to draw conclusions from the result:
1. make a plot like

    prompt type | cultural | musical term |
2/4             |          |              |
3/4             |          |              |      
4/4             |          |              |      
5/4             |          |              |      
6/8             |          |              |    

and each entry is the success rate

we could compare vertically and horizontally.

vertically tells us 1. user-experience bias: if model handles colloquial and culturally common descriptions better than formal musical terms, potentially disadvantaging musically trained users

horizontally tells us 2.cultural bias (if models collapse time signatures rare in Western pop music to 4/4, musical traditions that rely on non-4/4 rhythms are underrepresented)

Also, for each failure we record the failure type and calculate how many collpased to 4/4. 


how to evaluate correctness and calculate the success rate?
Use libora package to analyze generated audio files to track time signature. Also, I will manually verify and judge some samples to see if the time signature is correct. 

sample size for the experiment:
5 different prompts for each prompt type and each time signature.
perform each prompt 5 times
So, in total (5 time sigs) * (2 pormpt types) * (5 times) * (2 models) = 100 generation requests in total. 

what are model scope?
Suno and google

