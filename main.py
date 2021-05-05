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

MAX_INPUT_LENGTH = int(config['Settings']['MaxInputLength'])
DEVICE = config['Settings']['Device']
GC_EVERY_TIME = config['Settings']['UseGC'] == "yes"
HALF_PRECISION = config['Settings']['HalfPrecision'] == "yes"

print("Model path: " + modelpath)

def runFlask(pipe):
    webserver.pipe = pipe
    webserver.app.run(host="localhost", port="31013", debug=False)

def generateText(data):
    global model, tokenizer, mainpipe
    with torch.cuda.amp.autocast(enabled=True):
        with torch.no_grad():
            maxInputLength = MAX_INPUT_LENGTH - int(data['responseLength']) # This is how much of the context we can use based on the requested response size
            print("Generating...")
            t1 = time.time()
            input_ids = tokenizer(data['context'], return_tensors="pt").input_ids
            #print("input_ids", input_ids)
            if input_ids.size()[1] > maxInputLength:
                input_ids = input_ids.narrow(1, -maxInputLength, maxInputLength)
            input_ids = input_ids.to(DEVICE)
            #print("Resized tokens:", tokenizer.decode(input_ids[0]))
            #print("len", len(input_ids[0]))
            gen_tokens = model.generate(input_ids, do_sample=True, temperature=float(data['cTemperature']),
                                        max_length=min(MAX_INPUT_LENGTH, input_ids.size()[1] + int(data['responseLength'])),
                                        num_return_sequences=int(data['numResponses']), top_p=float(data['top_p']), top_k=int(data['top_k']),
                                        num_beams=int(data['num_beams']), repetition_penalty=float(data['repetition_penalty']))
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
            input_ids = tokenizer(data['context'], return_tensors="pt").input_ids
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
                #print(sampled_list[i], int(sampled[0][i]), float(softmax[0][int(sampled[0][i])]))
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

    if bool(wsdata['token_mode']):
        mainpipe.send(generateTokens(wsdata))
    else:
        mainpipe.send(generateText(wsdata))
