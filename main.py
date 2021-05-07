import sys, time, gc, configparser
import webserver, multiprocessing # The webserver frontend bits
from transformers import AutoModelForCausalLM, AutoTokenizer # The transformer bits
import torch


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

    if memory != "" and text != "":
        totalIDs = torch.cat( (memoryIDs[0], textIDs[0]) )
    elif memory == "" and text != "":
        totalIDs = textIDs[0]
    elif memory != "" and text == "":
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
            #print(input_ids)
            gen_tokens = model.generate(input_ids, do_sample=True, temperature=data['cTemperature'],
                                        max_length=min(MAX_INPUT_LENGTH, input_ids.size()[1] + data['responseLength']),
                                        num_return_sequences=data['numResponses'], top_p=data['top_p'], top_k=data['top_k'],
                                        num_beams=data['num_beams'], repetition_penalty=data['repetition_penalty'])
            sequences = []
            strsequences = []
            for x in gen_tokens:
                sequences.append(tokenizer.decode(x[-int(data['responseLength']):]))
            for x in sequences:
                new = "".join(x)
                strsequences.append(new)
            response = {
                "responses": strsequences,
                "topk_tokens": [],
                "softmax_tokens": []
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
                "responses": [],
                "topk_tokens": topk_list,
                "softmax_tokens": sampled_list,
                "sampled_probs": sampled_probs,
                "topk_probs": topk_probs
            }
            t2 = time.time()
            print("Finished generating tokens in", t2-t1)
            return response;
        
# Load the language model
print("Loading model at", modelpath)
t1 = time.time()
model = AutoModelForCausalLM.from_pretrained(modelpath)
t2 = time.time()
if HALF_PRECISION:
    model.to(torch.float16)
model.to(DEVICE)
model.eval()
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
        if wsdata['token_mode']:
            mainpipe.send(generateTokens(wsdata))
        else:
            mainpipe.send(generateText(wsdata))
