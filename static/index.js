var generating = false;
var generated = 0;
var gennumber = 0;
var tokenmode = false;

// This nightmare is what allows responses to be clicked on to retrieve portions, rather than the entire text
function updateResponsesWithListeners() {
    document.querySelectorAll('.response').forEach(el => {
	let fulltext = el['innerText'];
	let fulltextspan = document.createElement('span');
	fulltextspan.setAttribute("class", "fulltext");
	fulltextspan.textContent = fulltext;
        let characters = el['innerText'].split('');
        el.innerHTML = '';
	el.appendChild(fulltextspan);

        characters.forEach(char => {
	    let span = document.createElement('span');
	    span.innerText = char;
	    span.addEventListener('click', function () {
                let position = 0;
                let el = this;
                while (el.previousSibling !== null) {
		    position++;
		    el = el.previousSibling;
                }
		document.getElementById("textPanel").value += this.parentElement.getElementsByClassName('fulltext')[0].textContent.slice(0, position);
		clearResponses();
		generated = 0;
		gennumber++;
		if (isPickgenOn()) {
		    requestGenerate();
		}
		autoscrolldown();
	    });
	    el.appendChild(span);
        });
    });
}

function jsonifyData() {
    var g = function( id ) { return document.getElementById( id ).value; };

    var msg = {context: g("textPanel"), cTemperature: g("cTemperature"), numResponses: g("numResponses"), responseLength: g("responseLength"),
	       top_p: g("top_p"), top_k: g("top_k"), num_beams: g("num_beams"), repetition_penalty: g("repetition_penalty"),
	       memory: g("memory"), note: g("note"), share: g("share"), noteLinesBack: g("noteLinesBack"),
	       repetition_penalty_range: g("repetition_penalty_range"), repetition_penalty_slope: g("repetition_penalty_slope"),
	       token_mode: tokenmode
	      }
    return msg;
}

function requestGenerate() {
    if (generating) return;
    
    var json = jsonifyData();
    var data = JSON.stringify(jsonifyData());
    if (json.context == "" && json.memory == "" && json.note == "") {
	alert("No input! Write some text, or put something in memory.");
    }
    else {
	var xhr = new XMLHttpRequest();
	xhr.open("POST", "/gen", true);
	xhr.setRequestHeader("Content-Type", "application/json; charset=utf-8");
	xhr.onreadystatechange = function() {
	    if (this.readyState == 4 && this.status == 200) {
		verifyResponses(JSON.parse(this.responseText));
	    }
	}
	document.getElementById("genButton").classList.add("waiting");
	generating = true;
	xhr.send(data);
    }
}

function checkForResponses() {
    var status = 0;
    var r = null;
    fetch("/check")
	.then(response => verifyResponse(response));
    
}

function verifyResponses(data) {
    createResponses(data.responses);
    createTokens(data);
    generated ++;
    var toGen = document.getElementById("autogen").value;
    if (generated < toGen && generating == true) {
	generating = false;
	requestGenerate();
    }
    else {
	generating = false;
	document.getElementById("genButton").classList.toggle("waiting");
	generated = 0;
    }
}


function createResponses(responses) {
    if (responses.length == 0) return; // For when token mode is on
    var bp = document.getElementById("bottomPanel");
    for (const r of responses) {
	var nr = document.createElement("div");
	nr.className = "response";
	nr.textContent = r
	bp.prepend(nr);
    }
    updateResponsesWithListeners();
}

function clearResponses() {
    var responses = document.getElementsByClassName("response");
    while (responses[0]) {
	responses[0].parentNode.removeChild(responses[0]);
    }
    var tt = document.getElementById("tokenTable");
    if (tt != undefined) {
	tt.parentNode.removeChild(tt);
    }
}

function tokenClicked() {
    document.getElementById("textPanel").value += this.textContent;
    clearResponses();

    if (isPickgenOn()) {
	requestGenerate();
    }
    autoscrolldown();
}

function createTokens(jsonData) {
    if (jsonData.softmax_tokens.length == 0) return;
    var oldTable = document.getElementById("tokenTable");
    if (oldTable != undefined) {
	tokenTable.parentNode.removeChild(oldTable);
    }
    var tp = document.getElementById("tokenPanel");
    var ttable = document.createElement("table");
    ttable.id = "tokenTable";
    tp.appendChild(ttable);
    var tr1 = document.createElement("tr");
    ttable.appendChild(tr1);
    var th1 = document.createElement("th");
    th1.textContent = "TopK";
    var th2 = document.createElement("th");
    th2.textContent = "Sampled";
    tr1.appendChild(th1);
    tr1.appendChild(th2);

    // Softmaxing
    topk_probs = softmax(jsonData.topk_probs)
    sampled_probs = softmax(jsonData.sampled_probs)

    // Refactoring this will happen. later.
    for (var x = 0; x < jsonData.topk_tokens.length; x ++) {
	var row = document.createElement("tr");
	ttable.appendChild(row);
	var td1 = document.createElement("td");
	td1.textContent = jsonData.topk_tokens[x];
	td1.addEventListener('click', tokenClicked);
	row.appendChild(td1);
	var td1bar = document.createElement("div");
	td1bar.classList.add("percentbar");
	td1bar.style.height = (topk_probs[x]*100)+"%";
	td1.appendChild(td1bar);
	
	var td2 = document.createElement("td");
	td2.textContent = jsonData.softmax_tokens[x];
	td2.addEventListener('click', tokenClicked);
	row.appendChild(td2);
	var td2bar = document.createElement("div");
	td2bar.classList.add("percentbar");
	td2bar.style.height = (sampled_probs[x]*100)+"%";
	td2.appendChild(td2bar);
    }
}

//Standard path to hiding other UI elements to display a particular one afterward.
function hideUI() {
    document.getElementById("options").classList.add("hidden");
    document.getElementById("fullelement").classList.add("hidden");
    document.getElementById("memorywindow").classList.add("hidden");
}

function toggleOptions() {
    options = document.getElementById("options");
    if (options.classList.contains("hidden")) {
	hideUI();
	options.classList.remove("hidden");

    }
    else {
	hideUI();
	document.getElementById("fullelement").classList.remove("hidden");
    }
}

//a dirty copypaste; if we're going to have so many windows we should generalize this soon
function toggleMemorywindow() {
    memorywindow = document.getElementById("memorywindow");
    if (memorywindow.classList.contains("hidden")) {
	hideUI();
	memorywindow.classList.remove("hidden");

    }
    else {
	hideUI();
	document.getElementById("fullelement").classList.remove("hidden");
    }
}

function isPickgenOn() {
    var e = document.getElementById("pickgenButton")
    return e.classList.contains("activated")
}

function pickgenToggle() {
    var e = document.getElementById("pickgenButton");
    e.classList.toggle("activated");
}

function autoscrolldown() {
    var w = document.getElementById("textPanel");
    w.scrollTop = w.scrollHeight;
}

function swapGenerateToTokens() {
    document.getElementById("swapmodeButton").classList.toggle("activated");
    tokenmode = !tokenmode;
}

// src: https://gist.github.com/cyphunk/6c255fa05dd30e69f438a930faeb53fe#gistcomment-3649882
// Client-side softmax to shave just a little demand off the server.
function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const denom = scores.reduce((a, b) => a + b);
    return scores.map(s => s / denom);
}

document.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById("genButton").addEventListener("click", requestGenerate);
    
    document.getElementById("optionsButton").addEventListener("click", toggleOptions);
    document.getElementById("exitOptionsButton").addEventListener("click", toggleOptions);

    document.getElementById("memorywindowButton").addEventListener("click", toggleMemorywindow);
    document.getElementById("exitMemorywindowButton").addEventListener("click", toggleMemorywindow);
    
    document.getElementById("clearButton").addEventListener("click", clearResponses);
    document.getElementById("pickgenButton").addEventListener("click", pickgenToggle);
    document.getElementById("swapmodeButton").addEventListener("click", swapGenerateToTokens);

    document.addEventListener('keydown', function(event) {
	if (event.ctrlKey && event.key === 'e') {
	    requestGenerate();
	}
    });
})

