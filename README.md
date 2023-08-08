# Lunk but looped

This is an extremely rough MVP hack of the  LUNK script written by (authorname I forget). 

## Thought process
The basic idea of LUNK is that instead of simply merging all the weights of two models with the same architecture using a certain blend ratio, iterating through every parameter in the model and having a chance of the parameter being merged or not can better preserve the distinct qualities and characteristics of both models, if you're lucky. But this approach has a problem: since there's a random chance of the parameters being merged or not, you need to be quite fortunate to create a great model from a merge, like Hugin.

What's one way to deal with bad luck? Try again until you get good luck! 

## What this does

This ungodly, ugly hack takes the eleuther AI evaluation harness, the lunk script, and crudely wires them together so that the lunk script will make `n` different lunks and pick the one with the best average score according to the tests chosen for the benchmark. So that a huge amount of memory isn't consumed if you pick a high `n`, this script will compare the first two lunks made after lunking twice, and delete the worse one.

## Usage:

run

```
python lunk.py 2
```

To start up the Gradio UI. `2` means that you are going to create 2 lunks and pick the best one; you can change this to any number you want.
To install packages, you might want to create a new virtual environment and install python 3.9, then cd into TODO !EA add dirname and type `pip install .` to install all the many packages required by the eleuther harness.

This script is basically untested (I only tested that it does, in fact, loop properly, by merging a model into itself) and comes with a number of limitations, as it is an mvp:

1. **Mishmash of command line arguments and Gradio.** I don't know Gradio, so the number of times to lunk is currently supplied by sys.argv in lunk.py
2. **Bit tricky to change the tests used.** You need to go into lunk.py and change the arguments to the evaluate_model function to a string of the following format: "test1,test2,test3,test4". This should obviously be done from Gradio instead, but, mvp.
3. **Very specific requirements for directory structure.** I was in a rush to get this working and so stripped out a fair bit of customization. The model you're outputting to will now always be in the same directory as the lunk.py script. The directory structure of the repo must generally be well preserved unless you know what you're doing. converted_model_{somenumber} will be the one you want, after the script finishes
4. **Sharp edges everywhere.** This was crudely adapted within the space of a single day by someone who barely knew what they were doing (me) and so is both unpolished and untested
5. **Jesus Christ it takes a long time to run.** I mean it, this is something you should probably leave going overnight if you're using it for production purposes. I was testing on Pythia 460M (decidedly smaller than basically any useful model) and it took over 8 minutes to evaluate a single model on hellaswag alone. And hellaswag is far from the largest of the eval datasets. And you're doing this n times. Lunking itself is fast but evaluating models is not. So be prepared for long runtimes.
6. **Does it actually work?** I ran this on a small model and it seemed to go through all the steps right, but since I was merging the model with itself (so it would for sure have the same architecture) there was obviously no change. Theoretically all the functions have the right arguments but I might've made a stupid mistake somewhere and so this needs more testing.
7. **Only as good as the benchmarks it runs.** Multiple papers have mentioned current benchmarks as being insufficient for judging LLMs, and that writing quality is not really possible to evaluate, at least not yet. Because this script uses the same -- flawed -- benchmarks to pick the best LUNK, it cannot yet be optimized for writing quality.

At least it doesn't use the GPU that much (the lunks only have to forward pass, I think). But it does use one more model's worth of memory.