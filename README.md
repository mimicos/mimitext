# mimitext

## Overview

Mimitext is an interface to Huggingface's transformers library, providing a proof-of-concept user interface for the purposes of inference. It generates text based on the text you give it, with a number of configurable options.

Possibly unique to mimitext is that it also allows you to peek at the possible next tokens for a given input. For instance, you can view the TopK and SoftMax results from attempting to move the model forward. These tokens can be added to the text-so-far with a click. Of course, conventional string generation (currently provided by `generate` is still available.

When text is generated normally, the user can click on a letter in the possible result to add the text *up to* that letter, rather than being forced to accept the whole result. This should allow an easier way to move forward with less manual editing.
