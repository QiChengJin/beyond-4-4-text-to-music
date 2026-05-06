what is time signature?
When you listen to music, you can usually feel a repeating pulse — like a heartbeat. Time signature tells you how many of those pulses are grouped together before the pattern repeats.

The easiest way to feel it: try counting along to a song.

Most pop music: you count 1-2-3-4, 1-2-3-4 → that's 4/4
A waltz: you count 1-2-3, 1-2-3 → that's 3/4
An Irish jig: you count 1-2-3-4-5-6, 1-2-3-4-5-6 (in two bouncy groups) → that's 6/8
Mission Impossible theme: you count 1-2-3-4-5, 1-2-3-4-5 → that's 5/4, and it sounds slightly "off" because 5 doesn't divide evenly
Why does this matter for our project?
AI music models are trained mostly on Western pop music, which is overwhelmingly 4/4. We suspect they struggle — or outright fail — when asked to generate music in other time signatures. That's the bias we're testing. 


what we want to do for the project ?
https://www.reddit.com/r/SunoAI/comments/1po4jlu/54_time/
Motivated by the online community and my personal observation using Suno, we noticed the model is more likely to succeed when told “make a waltz” than when told “make a song in ¾ time signature”, even though these describe the exact same rhythm. Also, complex time signatures like 5/4 are hard to generate with output always collapsing to 4/4, a dominant time signature in western pop music. We plan to investigate two fairness issues through a case study of text-to-music generation with different time signatures with different prompts: 1. user-experience bias: if model handles colloquial and culturally common descriptions better than formal musical terms, potentially disadvantaging musically trained users). 2. cultural bias (if models collapse time signatures rare in Western pop music to 4/4, musical traditions that rely on non-4/4 rhythms are underrepresented). We plan to present our findings as a blog post or a collab notebook. 


what is my prompt design?
We have 50 prompts in total in prompt.md
each time signature has 5 pairs of prompts:
1. cultural prompt: describe the music using genre name or cultural reference. 
2. formal prompt: same description but use the time signature natation explicitly instead.





e.g. how to generate one song using Suno?
(there is an pdf version of instruction under /instruction folder)
1. Go to Suno.com and log in
2. click create
3. paste the prompt written in the prompt.md into the text box
4. click generate -> Suno will produce two versions -> download both
5. rename the file following this convention {time_sig nominator}_{time_sig denominator}_{prompt_type}_{prompt_num}_{run_num}.mp3

e.g.: 3_4_formal_1_1.mp3





the workflow:

for each prompt in prompt.md, repeat 5 times:
    1. copy-paste prompt -> 
    2. paste into Suno -> generate -> download (mp3/wav)
    3. rename the file using prompt_id
    4. place in the audio/Suno/ folder

when done with a batch, push to github

each Suno account has 50 credits daily, you might want to switch account to generate more songs. 










