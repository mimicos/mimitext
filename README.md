# mimitext

## Overview

Mimitext is an interface to Huggingface's transformers library, providing a proof-of-concept user interface for the purposes of inference. It generates text based on the text you give it, with a number of configurable options.

Possibly unique to mimitext is that it also allows you to peek at the possible next tokens for a given input. For instance, you can view the TopK and SoftMax results from attempting to move the model forward. These tokens can be added to the text-so-far with a click. Of course, conventional string generation (currently provided by `generate` is still available.

When text is generated normally, the user can click on a letter in the possible result to add the text *up to* that letter, rather than being forced to accept the whole result. This should allow an easier way to move forward with less manual editing.

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
