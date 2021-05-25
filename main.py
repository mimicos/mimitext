import sys, time, gc, configparser, inspect
import webserver, multiprocessing # The webserver frontend bits
from transformers import AutoModelForCausalLM, AutoTokenizer # The transformer bits
import torch
import torch.cuda.comm

if len(sys.argv) < 2:
    print("")
    print("Arguments: main.py <path to model>")
    print("Example: python3 main.py /home/user/gpt2")
    print("A model directory typically has a config.json and pytorch_model.bin file.")
    print(" Exiting ...")
    exit()

config = configparser.ConfigParser()
config.read("config.ini")

running = True
modelpath = sys.argv[1]
model = None
tokenizer = None

BIND_ADDRESS = config['Settings']['BindAddress']
BIND_PORT = int(config['Settings']['BindPort'])
MAX_INPUT_LENGTH = int(config['Settings']['MaxInputLength'])
DEVICE = config['Settings']['Device']
GC_EVERY_TIME = config['Settings']['UseGC'] == "yes"
HALF_PRECISION = config['Settings']['HalfPrecision'] == "yes"

print("Model path: " + modelpath)

def runFlask(pipe):
    webserver.pipe = pipe
    webserver.app.run(host=BIND_ADDRESS, port=BIND_PORT, debug=False)

# Preprocess the json data message to ensure all values are present and reasonable
def dataPreprocess(data):
    try:
        data['validRequest'] = True
        data['cTemperature'] = float(data['cTemperature']) if 'cTemperature' in data else 0.5
        data['responseLength'] = int(data['responseLength']) if 'responseLength' in data else 30
        data['numResponses'] = int(data['numResponses']) if 'numResponses' in data else 1
        data['top_p'] = float(data['top_p']) if 'top_p' in data else 1
        data['top_k'] = int(data['top_k']) if 'top_k' in data else 0
        data['num_beams'] = int(data['num_beams']) if 'num_beams' in data else 1
        data['repetition_penalty'] = float(data['repetition_penalty']) if 'repetition_penalty' in data else 1.0
        data['token_mode'] = bool(data['token_mode']) if 'token_mode' in data else False
        data['memory'] = data['memory'] if 'memory' in data else ""
        data['note'] = data['note'] if 'note' in data else ""
        data['noteLinesBack'] = int(data['noteLinesBack']) if 'noteLinesBack' in data else 3
        data['share'] = float(data['share']) if 'share' in data else .75
        data['repetition_penalty_range'] = int(data['repetition_penalty_range']) if 'repetition_penalty_range' in data else 300
        data['repetition_penalty_slope'] = float(data['repetition_penalty_slope']) if 'repetition_penalty_slope' in data else 3.33
    except KeyError:
        data['validRequest'] = False
        
    return data

def assembleContext(responseLength, text, memory, note, noteLinesBack = 3, share=.75):
    #This value is what we have to split between input text and other features, like the "memory"
    maxTotalInput = MAX_INPUT_LENGTH - responseLength
    maxMemorySize = int(maxTotalInput * (1 - share))
    maxTextSize = int(maxTotalInput * share)

    alteredText = text.splitlines()
    alteredText.insert(-noteLinesBack, note)
    alteredText = "\n".join(alteredText)

    textIDs = tokenizer(alteredText, return_tensors="pt").input_ids
    memoryIDs = tokenizer(memory, return_tensors="pt").input_ids
    if memoryIDs.size()[1] > maxMemorySize:
        memoryIDs = memoryIDs.narrow(1, -maxMemorySize, maxMemorySize)

    # Whatever you don't use for memory is more room for the text itself
    adjustedMaxTextSize = (maxTextSize + maxMemorySize - memoryIDs.size()[1])
    
    if textIDs.size()[1] > adjustedMaxTextSize:
        textIDs = textIDs.narrow(1, -adjustedMaxTextSize, adjustedMaxTextSize)

    # The logic here could be cleaned up a bit - I'm not sure if I want to keep
    # notes implemented this way (added to the text) rather than treating them differently
    if memory != "" and (text != "" or note != ""):
        totalIDs = torch.cat( (memoryIDs[0], textIDs[0]) )
    elif memory == "" and (text != "" or note != ""):
        totalIDs = textIDs[0]
    elif memory != "" and (text == "" and note == ""):
        totalIDs = memoryIDs[0]

    # Useful diagnostic prints:
    #print(totalIDs.unsqueeze(0), totalIDs.size())
    #print(tokenizer.decode(totalIDs))
    #print(maxTotalInput, maxMemorySize, maxTextSize, share, memoryIDs.size(), textIDs.size(), totalIDs.size(), maxTextSize + maxMemorySize - memoryIDs.size()[1])
    return totalIDs.unsqueeze(0)
    
def generateText(data):
    global model, tokenizer, mainpipe
    with torch.cuda.amp.autocast(enabled=True):
        with torch.no_grad():
            maxInputLength = MAX_INPUT_LENGTH - int(data['responseLength']) # This is how much of the context we can use based on the requested response size
            print("Generating...")
            t1 = time.time()

            input_ids = assembleContext(data['responseLength'], data['context'], data['memory'], data['note'], data['noteLinesBack'], data['share'])
            #input_ids = tokenizer(data['context'], return_tensors="pt").input_ids
            #if input_ids.size()[1] > maxInputLength:
            #    input_ids = input_ids.narrow(1, -maxInputLength, maxInputLength)

            input_ids = input_ids.to(DEVICE)

            # Check for finetuneanon's fork of transformers, which has extra features--
            # *or* this will work if those features make it to mainline (and keep the same names)
            if 'repetition_penalty_range' in inspect.signature(model.generate).parameters:
                print(" repetition_penalty range and slope available")
                gen_tokens = model.generate(input_ids,
                                            do_sample=True,
                                            temperature=data['cTemperature'],
                                            max_length=min(MAX_INPUT_LENGTH,
                                                           input_ids.size()[1] + data['responseLength']),
                                            num_return_sequences=data['numResponses'],
                                            top_p=data['top_p'], top_k=data['top_k'],
                                            num_beams=data['num_beams'],
                                            repetition_penalty=data['repetition_penalty'],
                                            repetition_penalty_range = data['repetition_penalty_range'],
                                            repetition_penalty_slope = data['repetition_penalty_slope']
                )
            # else fall back to the transformers 4.5.1 style call;
            # is there a way to do this within one function call attempt?
            else:
                print(" repetition_penalty range and slope NOT available")
                gen_tokens = model.generate(input_ids,
                                            do_sample=True,
                                            temperature=data['cTemperature'],
                                            max_length=min(MAX_INPUT_LENGTH,
                                                           input_ids.size()[1] + data['responseLength']),
                                            num_return_sequences=data['numResponses'],
                                            top_p=data['top_p'], top_k=data['top_k'],
                                            num_beams=data['num_beams'],
                                            repetition_penalty=data['repetition_penalty']
                )
                
            sequences = []
            strsequences = []
            for x in gen_tokens:
                sequences.append(tokenizer.decode(x[-int(data['responseLength']):]))
            for x in sequences:
                new = "".join(x)
                strsequences.append(new)
            response = {
                "responses": strsequences,
            }
            t2 = time.time()
            print("Finished in {} seconds".format(t2-t1))
            return response;

def generateTokens(data):
    global model, tokenizer, mainpipe
    with torch.cuda.amp.autocast(enabled=True):
        with torch.no_grad():
            t1 = time.time()
            maxInputLength = MAX_INPUT_LENGTH - 1 # (we just want one token)
            print("Generating tokens...")
            #input_ids = tokenizer(data['context'], return_tensors="pt").input_ids
            input_ids = assembleContext(data['responseLength'], data['context'], data['memory'], data['note'], data['noteLinesBack'], data['share'])
            if input_ids.size()[1] > maxInputLength:
                input_ids = input_ids.narrow(1, -maxInputLength, maxInputLength)
            input_ids = input_ids.to(DEVICE)
            a = model(input_ids)
            logits = a.logits[:, -1, :]
            topk_results = torch.topk(logits, k=100)
            topk_tolist = topk_results[1].tolist()[0]
            topk_list = tokenizer.batch_decode(topk_tolist)
            if (float(data['cTemperature']) > 0):
                logits = logits / float(data['cTemperature'])
            softmax = torch.nn.functional.softmax(logits, dim=-1)
            sampled = torch.multinomial(softmax, num_samples=100)
            sampled_list = tokenizer.batch_decode(sampled.tolist()[0])

            # For examining the actual probabilities - a work in progress
            topk_probs = []
            sampled_probs = []
            for i in range(len(sampled_list)):
                sampled_probs.append(float(logits[0][int(sampled[0][i])]))
                topk_probs.append(float(logits[0][topk_tolist[i]]))

            response = {
                "topk_tokens": topk_list,
                "softmax_tokens": sampled_list,
                "sampled_probs": sampled_probs,
                "topk_probs": topk_probs
            }
            t2 = time.time()
            print("Finished generating tokens in", t2-t1)
            return response;
        
#Define a new forward pass
def new_forward(
    self,
    input_ids=None,
    past_key_values=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    global breakmodel

    if breakmodel:
        
        global ram_blocks
        
        if not hasattr(self, 'extrastorage'):
            import copy
            setattr(self,"extrastorage",{})
            self.wte.to("cuda")
            self.wpe.to("cuda")
            self.ln_f.to("cuda")
            torch.cuda.empty_cache()
            for i in range(ram_blocks):
                self.h[i].to("cpu")
                self.extrastorage[i] = copy.deepcopy(self.h[i])
                smalltensor = torch.tensor(0).to("cuda")
                for param1 in self.h[i].parameters():
                    param1.data = smalltensor
                self.h[i].to("cuda")

            for i in range(ram_blocks,len(self.h)):
                self.h[i].to("cuda")

            for param in self.wte.parameters():
                param.requires_grad = False
            for param in self.wpe.parameters():
                param.requires_grad = False

            for i in range(len(self.h)):
                for param in self.h[i].parameters():
                    param.requires_grad = False
            for param in self.ln_f.parameters():
                param.requires_grad = False
            for i in range(ram_blocks):
                for param in self.extrastorage[i].parameters():
                    param.requires_grad = False
                    param.data.pin_memory()
            torch.cuda.empty_cache()

            for param1,param2 in zip(self.h[0].parameters(),self.extrastorage[0].parameters()):
                param1.data = param2.data.to("cuda", non_blocking=False)
            self.h[len(self.h)-1].to("cuda", non_blocking=False)

            for param1,param2 in zip(self.h[ram_blocks-1].parameters(),self.extrastorage[ram_blocks-1].parameters()):
                param1.data = param2.data.to("cuda", non_blocking=False)
            self.h[len(self.h)-1].to("cuda", non_blocking=False)


    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values[0][0].size(-2)

    device = input_ids.device if input_ids is not None else inputs_embeds.device
    if position_ids is None:
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    # Attention mask.
    if attention_mask is not None:
        assert batch_size > 0, "batch_size has to be defined and > 0"
        global_attention_mask = attention_mask.view(batch_size, -1)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        global_attention_mask = global_attention_mask[:, None, None, :]

        # Since global_attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        global_attention_mask = global_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        global_attention_mask = (1.0 - global_attention_mask) * -10000.0
    else:
        global_attention_mask = None

    # Local causal attention mask
    batch_size, seq_length = input_shape
    full_seq_length = seq_length + past_length
    local_attention_mask = GPTNeoAttentionMixin.create_local_attention_mask(
        batch_size, full_seq_length, self.config.window_size, device, attention_mask
    )

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x num_heads x N x N
    # head_mask has shape n_layer x batch x num_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.num_layers)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)
    position_embeds = self.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds

    if token_type_ids is not None:
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)

    output_shape = input_shape + (hidden_states.size(-1),)

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    if breakmodel :
        copystream = torch.cuda.Stream(device=0,priority = -1)

    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

        if breakmodel :
            if i in range(ram_blocks):
                index1 = (i+1)%ram_blocks
                for param1,param2 in zip(self.h[index1].parameters(),self.h[(i-1)%ram_blocks].parameters()):
                    param1.data = param2.data
                for param1,param2 in zip(self.h[index1].parameters(),self.extrastorage[index1].parameters()):
                    with torch.cuda.stream(copystream):
                        torch.cuda.comm.broadcast(param2.data,out = [param1.data])                    

        attn_type = self.config.attention_layers[i]
        attn_mask = global_attention_mask if attn_type == "global" else local_attention_mask

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if getattr(self.config, "gradient_checkpointing", False) and self.training:

            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                None,
                attn_mask,
                head_mask[i],
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attn_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        if breakmodel:
            if i in range(ram_blocks):
                torch.cuda.synchronize()

    if breakmodel:
        del copystream

    torch.cuda.empty_cache()

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(*output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


if __name__ == '__main__':
    # Load the language model
    print("Loading model at", modelpath)
    t1 = time.time()
    model = AutoModelForCausalLM.from_pretrained(modelpath)
    t2 = time.time()
    if HALF_PRECISION:
        model.to(torch.float16)
    
    modified_loading = True
    
    if not modified_loading:
        model.to(DEVICE)
        model.eval()
    else:
        #Partially move model to gpu
        model.eval().half().to("cpu")
        gc.collect()
        model.lm_head.to("cuda")
        model.transformer.wte.to("cuda")
        model.transformer.wpe.to("cuda")
        model.transformer.ln_f.to("cuda")
        # Number of blocks that will be on on RAM
        breakmodel = True
        ram_blocks = 7
        #Import libraries where forward pass is located
        from transformers import GPTNeoForCausalLM,GPTNeoModel
        from transformers.modeling_outputs import BaseModelOutputWithPast
        from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoAttentionMixin
        #Swap out the forward pass
        GPTNeoModel.forward = new_forward
        
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    print("Model+tokenizer loaded in", t2-t1, "seconds")

    # Initialize flask, the webserver providing the frontend interface
    mainpipe, webservpipe = multiprocessing.Pipe()
    webservprocess = multiprocessing.Process(target=runFlask, args=(webservpipe,))
    webservprocess.start()
    while running:
        if GC_EVERY_TIME:
            gc.collect()
            torch.cuda.empty_cache()
        wsdata = mainpipe.recv()
        wsdata = dataPreprocess(wsdata)

        if wsdata['validRequest']:
            # Tokens-only path
            responseData = {"responses": [], "topk_tokens": [], "softmax_tokens": [], "sampled_probs": [], "topk_probs": []}

            tokenData = generateTokens(wsdata)
            textData = {}
            if not wsdata['token_mode']:
                textData = generateText(wsdata)

            responseData.update(tokenData)
            responseData.update(textData)

            mainpipe.send(responseData)
