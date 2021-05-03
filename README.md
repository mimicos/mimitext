# mimitext

## Overview

Mimitext is an interface to Huggingface's transformers library, providing a proof-of-concept user interface for the purposes of inference. It generates text based on the text you give it, with a number of configurable options.

This is provided via **Flask** and runs on **localhost**. Despite being a simple webserver, mimitext is currently *not intended for more than one user at a time.* It will likely break horribly in that scenario. However, the webserver component means it can be easily run on a distant machine and operated from another. For this, I recommend *ssh -D*.

Possibly unique to mimitext is that it also allows you to peek at the possible next tokens for a given input. For instance, you can view the TopK and SoftMax results from attempting to move the model forward. These tokens can be added to the text-so-far with a click. Of course, conventional string generation (currently provided by `generate` is still available.

When text is generated normally, the user can click on a letter in the possible result to add the text *up to* that letter, rather than being forced to accept the whole result. This should allow an easier way to move forward with less manual editing.

## Sample Images

### "Normal" mode
[Normal Mode Example](demo/demo01.png)

When the user highlights a character it shows that, upon click, all text up to that point is transferred; the text after is deleted.

### "Token" mode
[Token Mode Example](demo/demo02.png)

Here a user can manually click on the next token to add to the text. This might be useful for when the model isn't generating very interesting responses on its own.

The interior shaded cells currently only represent the token's position out of 100. Ideally, it would be a useful representation of the relative probabilities.

## GUI
There are five buttons making up the webgui right now.

### Generate
This sends a request to the server, which is then sent to the model, via POST.

### Options
This displays the **Generation Options** seen below.

### Clear
This will erase both the possible responses (below the main text) and the list of possible tokens (in token mode).

### Pick-generate
When a possible response is clicked, or a token is clicked, the next generate request is made automatically (as if the user had clicked Generate immediately afterward.)

### Swap Modes
Alters the display between showing a list of possible generation responses or a list of possible tokens following the main text. In the so-called *token mode* requests for strings of text aren't made at all (and in the normal mode, requests for tokens aren't currently made either.) Note that looking at the next possible tokens is similar to a response length of 1: it's very fast.

## Generation Options
These are almost identical to the transformers `generate` call; the values from the interface are passed directly into it.

### Autogenerate
This simply repeats the generation request after one is returned. The number is how many repeats. For instance, setting it to three is identical to just hitting "Generate" three times in a row.

### Temperature
Randomness. 0.3~0.7 is a good area to try for many models. *This value impacts the result of the SoftMax column in the Token mode.*

### Response Length
The length (in tokens) to request from the model. These are sometimes words, parts of words, individual letters, or less.

### top_p (nucleus) sampling
TODO: describe this

### top_k sampling
Restricts the possible results to the x most likely tokens. 0 to disable.

### Repetition Penalty
Higher values penalize tokens that have appeared in the input more and more, encouraging the responses to feel new. This sometimes has an unintended effect, as many tokens show up many times (the, The, names, and so on.)
Currently this has no impact on the token list that's generated in token mode.

### num_beams
TODO: describe. Beam search.
