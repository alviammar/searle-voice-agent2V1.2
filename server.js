const express = require("express");
const path = require("path");
const fs = require("fs");
const crypto = require("crypto");
const { WebSocket, WebSocketServer } = require("ws");

const app = express();
// The UI may be hosted separately (for example on Netlify) while this
// always-on Node/WebSocket backend runs on Render/Railway/Fly/Cloud Run.
// No cookies or browser credentials are used, so a simple CORS policy is
// appropriate for the demo. Lock this to your production domain later.
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", process.env.CORS_ORIGIN || "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});
app.use(express.json({ limit: "15mb" })); // generous limit for base64 prescription photos
app.use(express.static(path.join(__dirname, "public")));

const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
// Sonnet stays the default deliberately — this is medical triage content,
// and a faster/lighter model is a real quality trade-off, not a free win.
// Override with CLAUDE_MODEL if you want to test something faster
// (e.g. a Haiku-tier model) — but review triage accuracy carefully before
// trusting it with real patients if you do.
const MODEL = process.env.CLAUDE_MODEL || "claude-sonnet-4-6";

// The system prompt (medicine catalog, Extor knowledge, doctor list, all the
// triage rules) is identical on every turn of a call — marking it cacheable
// means Claude skips reprocessing those ~12K characters after the first
// message, cutting the "thinking" pause on turns 2+. Below the ~1024-token
// minimum, caching wouldn't apply anyway, so this stays a plain string then.
function buildCachedSystem(systemText) {
  const text = systemText || "";
  if (text.length < 3000) return text; // rough chars-to-tokens safety margin
  return [
    { type: "text", text, cache_control: { type: "ephemeral" } }
  ];
}

const SONIOX_API_KEY = process.env.SONIOX_API_KEY;
// "tts-rt-v1" is what Soniox's own official reference implementation uses as
// default — confirmed from their real source code, not guessed.
const SONIOX_TTS_MODEL = process.env.SONIOX_TTS_MODEL || "tts-rt-v1";
// "Mina" is one of Soniox's built-in voices (steady, warm, clear per their own
// description) — any of their voices can speak any of their 60+ languages,
// Urdu included, since it's one unified model rather than per-language voices.
// Swap via SONIOX_TTS_VOICE once you've listed real options (see README).
const SONIOX_TTS_VOICE = process.env.SONIOX_TTS_VOICE || "Mina";
const SONIOX_LANGUAGE = "ur"; // Urdu — confirmed ISO code from Soniox's docs

// Uplift AI is Sania's PRIMARY speaking voice. Soniox remains the realtime
// listening/STT engine and an emergency TTS fallback only. Keep the Uplift
// API key server-side; never expose it in public/config.js or browser code.
const UPLIFT_API_KEY = process.env.UPLIFT_API_KEY;
const UPLIFT_TTS_URL = process.env.UPLIFT_TTS_URL || "https://api.upliftai.org/v1/synthesis/text-to-speech";
const UPLIFT_TTS_VOICE = process.env.UPLIFT_TTS_VOICE || "paediatrician";
const UPLIFT_TTS_FORMAT = process.env.UPLIFT_TTS_FORMAT || "MP3_22050_128";
const TTS_PROVIDER = (process.env.TTS_PROVIDER || "uplift").toLowerCase();
const ENABLE_SONIOX_TTS_FALLBACK = process.env.ENABLE_SONIOX_TTS_FALLBACK !== "0";


// --------------------------------------------------------------------------
// FAST EXTOR BUILD: local TTS cache + safe conversation-learning store.
// Medical facts never self-edit. Only caller wording / intent examples and
// fallback questions are logged so the approved Extor knowledge remains fixed.
// --------------------------------------------------------------------------
const DATA_DIR = path.join(__dirname, "data");
const TTS_CACHE_DIR = path.join(DATA_DIR, "tts-cache");
const LEARNING_LOG = path.join(DATA_DIR, "learning-events.jsonl");
const LEARNED_PHRASES_FILE = path.join(DATA_DIR, "learned-phrases.json");
const LEARNED_ANSWERS_FILE = path.join(DATA_DIR, "learned-answers.json");
fs.mkdirSync(TTS_CACHE_DIR, { recursive: true });
if (!fs.existsSync(LEARNED_PHRASES_FILE)) fs.writeFileSync(LEARNED_PHRASES_FILE, JSON.stringify({}, null, 2));
if (!fs.existsSync(LEARNED_ANSWERS_FILE)) fs.writeFileSync(LEARNED_ANSWERS_FILE, JSON.stringify([], null, 2));

function ttsCacheKey(text, tone) {
  const providerSig = TTS_PROVIDER === "uplift"
    ? `uplift|${UPLIFT_TTS_VOICE}|${UPLIFT_TTS_FORMAT}`
    : `soniox|${SONIOX_TTS_MODEL}|${SONIOX_TTS_VOICE}`;
  return crypto.createHash("sha1").update(`${providerSig}|${tone || "normal"}|${text}`).digest("hex");
}
function ttsCachePath(text, tone) { return path.join(TTS_CACHE_DIR, ttsCacheKey(text, tone) + ".mp3"); }
function ttsCacheMetaPath(text, tone) { return path.join(TTS_CACHE_DIR, ttsCacheKey(text, tone) + ".meta.json"); }
function logVoiceSource(callId, provider, voice, cached=false){
  console.log("\n------------------------------------------------------------");
  console.log(`[VOICE SOURCE] ${provider}${cached ? " (CACHE)" : ""}`);
  console.log(`[VOICE] ${voice || "unknown"}`);
  console.log(`[CALL] ${callId}`);
  console.log("------------------------------------------------------------\n");
}

const LEARN_DEDUPE_WINDOW_MS = 8000;
const recentLearningEvents = new Map();
function normalizeIntentName(value){
  const x=String(value||"").trim().toLowerCase();
  const aliases={unknown_extor:"other_extor",precautions:"other_extor",kidney_precautions:"kidney_liver",renal_precautions:"kidney_liver",liver_precautions:"kidney_liver",interaction:"other_medicines",interactions:"other_medicines",medicine_interaction:"other_medicines",drug_interaction:"other_medicines",dose:"dose_frequency",dosage_frequency:"dose_frequency",competitors:"competitor",alternatives:"competitor",cost:"price"};
  return aliases[x]||x;
}
function learningText(value){ return String(value||"").toLowerCase().replace(/[؟?۔.,!،;؛:()"']/g," ").replace(/\s+/g," ").trim(); }
function hasLearnAny(t,list){ return list.some(x=>t.includes(x)); }
function inferLearningIntent(text, proposedIntent){
  const t=learningText(text); let proposed=normalizeIntentName(proposedIntent);
  if(hasLearnAny(t,["kidney","renal","گردے","گردہ","gurday","gurda"])) return "kidney_liver";
  if(hasLearnAny(t,["liver","hepatic","جگر","jigar"])) return "kidney_liver";
  if(hasLearnAny(t,["pregnant","pregnancy","حمل","حاملہ","hamal"])) return "pregnancy";
  if(hasLearnAny(t,["breastfeed","breastfeeding","دودھ پلا","feeding baby","nursing"])) return "breastfeeding";
  if(hasLearnAny(t,["potassium","پوٹاشیم"])) return "potassium";
  if(hasLearnAny(t,["under 18","under18","18 years","18 سال","child","children","teen","بچہ","بچوں"])) return "under18";
  if(hasLearnAny(t,["twice","two times","2 times","2 tablets","two tablets","double dose","do baar","2 baar","دو بار","دو گولی","دن میں دو"])) return "dose_frequency";
  if(hasLearnAny(t,["price","cost","rate","قیمت","دام","روپے","pkr","kitne ki","kitni ki"])) return "price";
  if(hasLearnAny(t,["competitor","alternative","substitute","compare","comparison","versus","better","exforge","avsar","amlortan","amstan","dioplus","newday","valam","valtec"])) return "competitor";
  if(hasLearnAny(t,["missed dose","miss dose","dose miss","bhool","بھول"])) return "missed_dose";
  if(hasLearnAny(t,["side effect","side effects","سائیڈ ایفیکٹ","نقصان"])) return "side_effects";
  if(hasLearnAny(t,["chakkar","dizziness","dizzy","swelling","soojan","سوجن","headache","سر درد"]) && hasLearnAny(t,["mujhe","مجھے","having","feel","feeling","ho raha","ہو رہا","what should","kya kar"])) return "side_effect_help";
  if(hasLearnAny(t,["panadol","paracetamol","ibuprofen","brufen","other medicine","another medicine","koi aur dawa","ساتھ کون سی دوا","interaction"])) return "other_medicines";
  if(hasLearnAny(t,["stop extor","stop taking","band kar","بند کر","چھوڑ دوں"])) return "stopping";
  if(hasLearnAny(t,["co extor","co-extor","کو ایکسٹور"])) return "co_extor";
  if(hasLearnAny(t,["overdose","too many tablets","extra tablets","زیادہ گولیاں"])) return "overdose";
  if(hasLearnAny(t,["chest pain","سینے میں درد","difficulty breathing","سانس","faint","بے ہوش"])) return "emergency";
  if(hasLearnAny(t,["5/80","5 80","5/160","5 160","10/160","10 160","strength","power","variation","طاقت"])) return "strengths";
  if(hasLearnAny(t,["with food","without food","empty stomach","khane","کھانے","خالی پیٹ"])) return "food";
  if(hasLearnAny(t,["what time","when should","kab","کس وقت","کب لوں","morning","evening","night"])) return "timing";
  if(hasLearnAny(t,["how to take","how should i take","kaise loon","کیسے لوں","water","pani","پانی","milk","doodh","دودھ"])) return "how_to_take";
  if(hasLearnAny(t,["what is extor","extor kya hai","ایکسٹور کیا ہے","used for","kis liye","کس لیے"])) return "what_is";
  if(proposed && proposed!=="other_extor") return proposed;
  return "other_extor";
}
function isDuplicateLearningEvent(callId,text,intent,source){
  const key=[String(callId||""),learningText(text),String(intent||""),String(source||"")].join("|"); const now=Date.now(),prev=recentLearningEvents.get(key)||0; recentLearningEvents.set(key,now);
  if(recentLearningEvents.size>500){ for(const [k,ts] of recentLearningEvents) if(now-ts>60000) recentLearningEvents.delete(k); }
  return !!prev && (now-prev)<LEARN_DEDUPE_WINDOW_MS;
}
function addLearnedPhrase(intent,text){
  intent=normalizeIntentName(intent); const normalized=learningText(text); if(!intent||normalized.length<3)return;
  try{ const learned=JSON.parse(fs.readFileSync(LEARNED_PHRASES_FILE,"utf8")||"{}"); learned[intent]=Array.isArray(learned[intent])?learned[intent]:[]; if(!learned[intent].includes(normalized)){ learned[intent].push(normalized); learned[intent]=learned[intent].slice(-250); fs.writeFileSync(LEARNED_PHRASES_FILE,JSON.stringify(learned,null,2)); } }catch(e){}
}

app.post("/api/learn", (req, res) => {
  const evt = req.body || {};
  const safe = {
    ts: new Date().toISOString(),
    callId: String(evt.callId || ""),
    text: String(evt.text || "").slice(0, 500),
    intent: inferLearningIntent(evt.text, evt.intent ? String(evt.intent).slice(0, 80) : null),
    matched: !!evt.matched,
    usedClaude: !!evt.usedClaude,
    latencyMs: Number.isFinite(evt.latencyMs) ? evt.latencyMs : null,
    source: String(evt.source || (evt.usedClaude ? "CLAUDE" : "BUILT_IN")).slice(0, 40)
  };
  const duplicate = isDuplicateLearningEvent(safe.callId, safe.text, safe.intent, safe.source);
  if (!duplicate) { try { fs.appendFileSync(LEARNING_LOG, JSON.stringify(safe) + "\n"); } catch (e) {} }
  else console.log(`[learning-dedupe] skipped duplicate intent=${safe.intent} source=${safe.source}`);

  // Human-readable source flag for live training/debugging in Terminal.
  // CLAUDE_REQUEST is routing-only; CLAUDE_RESPONSE is the actual spoken reply.
  if (safe.source !== "CLAUDE_REQUEST") {
    const label = safe.source === "CLAUDE_RESPONSE" ? "CLAUDE" : safe.source;
    console.log("\n============================================================");
    console.log(`[RESPONSE SOURCE] ${label}`);
    if (safe.intent) console.log(`[INTENT] ${safe.intent}`);
    if (safe.text) console.log(`[QUESTION] ${safe.text}`);
    if (safe.latencyMs != null) console.log(`[RESPONSE TIME] ${safe.latencyMs} ms`);
    console.log("============================================================\n");
  }

  // Auto-learn ONLY wording for an already-approved Extor intent. Never learn
  // a new medical fact, off-topic classification, dose, or response from a caller.
  const APPROVED_LEARN_INTENTS = new Set([
    "what_is","strengths","dose_frequency","side_effects","side_effect_help","side_effect_medicine","other_medicines","missed_dose","how_to_take","timing",
    "food","pregnancy","breastfeeding","kidney_liver","potassium","under18","stopping","co_extor","overdose","emergency","purchase","price","competitor"
  ]);
  if (safe.matched && APPROVED_LEARN_INTENTS.has(safe.intent) && safe.text.length >= 3) {
    addLearnedPhrase(safe.intent, safe.text);
  }
  res.json({ ok: true });
});


function normalizeLearnText(value) {
  return String(value || "").toLowerCase()
    .replace(/[؟?۔.,!،;؛:]/g, " ")
    .replace(/\s+/g, " ").trim();
}

// Learned answers are a performance cache, not a self-editing medical source.
// Only answers produced by Claude under the locked Extor prompt can be saved,
// and only for a small approved set of general Extor intents.
app.post("/api/learn-answer", (req, res) => {
  const body = req.body || {};
  const allowed = new Set(["what_is", "strengths", "dose_frequency", "side_effect_help", "side_effects", "side_effect_medicine", "other_medicines", "timing", "food", "missed_dose", "how_to_take", "pregnancy", "breastfeeding", "kidney_liver", "potassium", "under18", "stopping", "co_extor", "overdose", "emergency", "price", "competitor", "purchase", "other_extor"]);
  const question = normalizeLearnText(body.question).slice(0, 500);
  const intent = inferLearningIntent(question, body.intent);
  const answer = String(body.answer || "").replace(/<state>[\s\S]*?<\/state>/g, "").trim().slice(0, 700);
  if (!allowed.has(intent) || question.length < 5 || answer.length < 8) return res.status(400).json({ok:false});
  // Do not persist clearly patient-specific numeric advice. Vitals can still be
  // handled live by Claude, but that answer should not become reusable.
  if (/\b\d{2,3}\s*(?:\/|over|by)\s*\d{2,3}\b/i.test(question) || /\b\d{2,3}\s*kg\b/i.test(question)) {
    return res.json({ok:true, skipped:"patient_specific"});
  }
  try {
    let rows = JSON.parse(fs.readFileSync(LEARNED_ANSWERS_FILE, "utf8") || "[]");
    if (!Array.isArray(rows)) rows = [];
    const existing = rows.find(r => r.intent === intent && r.question === question);
    const row = { intent, question, answer, updatedAt:new Date().toISOString(), hits: existing ? Number(existing.hits||0) : 0 };
    rows = rows.filter(r => !(r.intent === intent && r.question === question));
    rows.push(row);
    rows = rows.slice(-400);
    fs.writeFileSync(LEARNED_ANSWERS_FILE, JSON.stringify(rows, null, 2));
    addLearnedPhrase(intent, question);
    console.log(`[learning-promote] intent=${intent} question="${question.slice(0,120)}"`);
    return res.json({ok:true, intent});
  } catch (e) {
    return res.status(500).json({ok:false,error:e.message});
  }
});

app.get("/api/learned-answers", (req, res) => {
  try {
    const rows = JSON.parse(fs.readFileSync(LEARNED_ANSWERS_FILE, "utf8") || "[]");
    res.json({ok:true, answers:Array.isArray(rows) ? rows : []});
  } catch (e) { res.json({ok:true, answers:[]}); }
});

app.get("/api/learned-phrases", (req, res) => {
  let learned = {};
  try { learned = JSON.parse(fs.readFileSync(LEARNED_PHRASES_FILE, "utf8") || "{}"); } catch (e) {}
  res.json({ ok: true, learned });
});

app.get("/api/learning-summary", (req, res) => {
  let learned = {};
  try { learned = JSON.parse(fs.readFileSync(LEARNED_PHRASES_FILE, "utf8") || "{}"); } catch (e) {}
  let recent = [];
  try {
    const lines = fs.readFileSync(LEARNING_LOG, "utf8").trim().split("\n").filter(Boolean).slice(-50);
    recent = lines.map(x => { try { return JSON.parse(x); } catch (e) { return null; } }).filter(Boolean).reverse();
  } catch (e) {}
  let answers = [];
  try { answers = JSON.parse(fs.readFileSync(LEARNED_ANSWERS_FILE, "utf8") || "[]"); } catch (e) {}
  if (!Array.isArray(answers)) answers = [];
  const learnedAnswerCounts = {};
  for (const row of answers) { const k=String(row && row.intent || "unknown"); learnedAnswerCounts[k]=(learnedAnswerCounts[k]||0)+1; }
  const quality = { learnedAnswers:answers.length, learnedPhrases:Object.values(learned).reduce((n,v)=>n+(Array.isArray(v)?v.length:0),0), recentClaudeFallbacks:recent.filter(x=>x&&x.usedClaude).length, recentLearnedDbHits:recent.filter(x=>x&&String(x.source||"").includes("LEARNED_DB")).length, recentUnknown:recent.filter(x=>x&&(x.intent==="unknown_extor"||x.intent==="other_extor")).length };
  res.json({ ok: true, learnedPhraseCounts: Object.fromEntries(Object.entries(learned).map(([k,v]) => [k, Array.isArray(v) ? v.length : 0])), learnedAnswerCounts, quality, recent });
});

app.post("/api/chat", async (req, res) => {
  if (!ANTHROPIC_API_KEY) {
    return res.status(500).json({
      error: { message: "Server is missing the ANTHROPIC_API_KEY environment variable. Set it in your host's dashboard and restart." }
    });
  }

  const { system, messages } = req.body || {};
  if (!Array.isArray(messages)) {
    return res.status(400).json({ error: { message: "Request body must include a messages array." } });
  }

  try {
    const upstream = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 220,
        temperature: 0.3,
        system: buildCachedSystem(system),
        messages
      })
    });

    const data = await upstream.json();
    res.status(upstream.status).json(data);
  } catch (err) {
    res.status(502).json({ error: { message: "Could not reach Claude API: " + err.message } });
  }
});

// Streaming version — forwards Claude's Server-Sent Events straight through as they
// arrive, so the frontend can start speaking a sentence before the whole reply (and
// the trailing tracking JSON) has finished generating. Combined with a deliberately
// smaller max_tokens budget and lower temperature, this nudges the agent toward
// short, phone-friendly turns instead of long speeches. /api/chat above is kept
// as a non-streaming fallback.
app.post("/api/chat-stream", async (req, res) => {
  if (!ANTHROPIC_API_KEY) {
    res.writeHead(500, { "Content-Type": "application/json" });
    return res.end(JSON.stringify({ error: { message: "Server is missing the ANTHROPIC_API_KEY environment variable." } }));
  }

  const { system, messages } = req.body || {};
  if (!Array.isArray(messages)) {
    res.writeHead(400, { "Content-Type": "application/json" });
    return res.end(JSON.stringify({ error: { message: "Request body must include a messages array." } }));
  }

  const t0 = Date.now();
  console.log(`[timing] Claude request started (system prompt: ${(system||"").length} chars)`);

  try {
    const upstream = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 220,
        temperature: 0.3,
        system: buildCachedSystem(system),
        messages,
        stream: true
      })
    });

    console.log(`[timing] Claude HTTP connection established: +${Date.now() - t0}ms`);

    if (!upstream.ok || !upstream.body) {
      const errText = await upstream.text().catch(() => "");
      res.writeHead(upstream.status || 502, { "Content-Type": "application/json" });
      return res.end(JSON.stringify({ error: { message: "Claude stream error: " + errText } }));
    }

    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive"
    });

    let firstTokenLogged = false;
    const reader = upstream.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!firstTokenLogged) {
        firstTokenLogged = true;
        console.log(`[timing] Claude FIRST TOKEN received: +${Date.now() - t0}ms`);
      }
      res.write(value);
    }
    console.log(`[timing] Claude stream fully complete: +${Date.now() - t0}ms`);
    res.end();
  } catch (err) {
    try {
      res.writeHead(502, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: { message: "Could not reach Claude API: " + err.message } }));
    } catch (e) { /* headers likely already sent */ }
  }
});

app.get("/healthz", (req, res) => res.json({
  ok: true,
  hasClaudeKey: !!ANTHROPIC_API_KEY,
  hasSonioxKey: !!SONIOX_API_KEY,
  hasUpliftKey: !!UPLIFT_API_KEY,
  ttsProvider: TTS_PROVIDER,
  upliftVoice: UPLIFT_TTS_VOICE
}));

// Real-time voice over Soniox's WebSocket API — one dedicated connection per
// TTS request (see handleTTSRequest below). Built from Soniox's raw protocol
// documentation, not their SDK — the published @soniox/node package (v1.1.2)
// doesn't actually expose the TTS methods its own docs describe.
//
// NOTE: an earlier version of this shared ONE persistent WebSocket connection
// across every reply in a call (multiplexed by stream_id), to save the
// repeated handshake time. That was reverted after live evidence (the 🚩
// AUDIO INTERRUPTED flag) caught Soniox occasionally losing track of one
// stream's finish signal on the shared connection while other streams on
// that same connection kept working fine — a 15-second freeze mid-reply.
// Since replies are already one request each (not split per-sentence), the
// latency saved by ALSO sharing the connection across replies was small;
// not worth trading for that failure mode.
let streamIdCounter = 0; // monotonic — makes stream_id collisions impossible, unlike Date.now()+random alone

// Split long Urdu replies into small natural phrases for synthesis, but do NOT
// play them one-by-one in the browser. Each phrase is synthesized in parallel,
// buffered on the server, then concatenated into one MP3 response. This avoids
// the "talk -> silence -> talk -> silence" effect caused by a single very long
// real-time TTS stream starving the browser's playback buffer.
function splitTTSChunks(text, maxChars = 180) {
  const cleaned = String(text || "").replace(/\s+/g, " ").trim();
  if (!cleaned) return [];

  // Urdu/English sentence punctuation first. Keep punctuation attached so the
  // voice still gets natural prosody cues.
  const sentences = cleaned.match(/[^۔.!?؟]+[۔.!?؟]?/g) || [cleaned];
  const chunks = [];
  let current = "";

  function pushCurrent() {
    const v = current.trim();
    if (v) chunks.push(v);
    current = "";
  }

  for (const raw of sentences) {
    const sentence = raw.trim();
    if (!sentence) continue;

    if ((current + " " + sentence).trim().length <= maxChars) {
      current = (current ? current + " " : "") + sentence;
      continue;
    }
    pushCurrent();

    if (sentence.length <= maxChars) {
      current = sentence;
      continue;
    }

    // One unusually long sentence: split on commas / Urdu comma / semicolons.
    const phrases = sentence.split(/(?<=[،,;؛:])\s+/);
    for (const phraseRaw of phrases) {
      const phrase = phraseRaw.trim();
      if (!phrase) continue;
      if ((current + " " + phrase).trim().length <= maxChars) {
        current = (current ? current + " " : "") + phrase;
      } else {
        pushCurrent();
        // Absolute fallback for text with no useful punctuation.
        if (phrase.length > maxChars) {
          for (let i = 0; i < phrase.length; i += maxChars) {
            chunks.push(phrase.slice(i, i + maxChars).trim());
          }
        } else {
          current = phrase;
        }
      }
    }
  }
  pushCurrent();
  return chunks;
}

let activeLiveTTS = 0;
let prewarmRunning = false;
let liveCallActive = false;

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// One raw Soniox TTS attempt. A wrapper below retries transient socket failures.
function synthesizeTTSChunkOnce(text, speed, callId, chunkIndex, totalChunks, attempt) {
  return new Promise((resolve, reject) => {
    const streamId = "s" + Date.now() + "_" + (streamIdCounter++);
    const started = Date.now();
    const ws = new WebSocket("wss://tts-rt.soniox.com/tts-websocket", {
      handshakeTimeout: 10000
    });
    const buffers = [];
    let settled = false;
    let firstAudioLogged = false;

    const timeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      try { ws.terminate(); } catch (e) {}
      reject(new Error(`Soniox TTS chunk ${chunkIndex + 1}/${totalChunks} timed out (attempt ${attempt})`));
    }, 30000);

    function fail(message) {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      try { ws.terminate(); } catch (e) {}
      reject(new Error(message));
    }

    ws.addEventListener("open", () => {
      try {
        ws.send(JSON.stringify({
          api_key: SONIOX_API_KEY,
          stream_id: streamId,
          model: SONIOX_TTS_MODEL,
          language: SONIOX_LANGUAGE,
          voice: SONIOX_TTS_VOICE,
          audio_format: "mp3",
          speed
        }));
        ws.send(JSON.stringify({
          stream_id: streamId,
          text,
          text_end: true
        }));
      } catch (e) {
        fail(`Soniox TTS send failed on chunk ${chunkIndex + 1}/${totalChunks}: ${e.message}`);
      }
    });

    ws.addEventListener("message", (event) => {
      if (settled) return;
      let msg;
      try { msg = JSON.parse(event.data); } catch (e) { return; }

      if (msg.error_message || msg.error_type) {
        return fail("Soniox error: " + (msg.error_message || msg.error_type));
      }

      if (msg.audio && typeof msg.audio === "string") {
        if (!firstAudioLogged) {
          firstAudioLogged = true;
          console.log(`[timing] [call ${callId}] chunk ${chunkIndex + 1}/${totalChunks} first audio: +${Date.now() - started}ms (attempt ${attempt})`);
        }
        buffers.push(Buffer.from(msg.audio, "base64"));
      }

      if (msg.terminated === true) {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        try { ws.close(); } catch (e) {}
        const out = Buffer.concat(buffers);
        if (!out.length) return reject(new Error(`Soniox returned no audio for chunk ${chunkIndex + 1}/${totalChunks}`));
        console.log(`[timing] [call ${callId}] chunk ${chunkIndex + 1}/${totalChunks} complete: +${Date.now() - started}ms (${out.length} bytes, attempt ${attempt})`);
        resolve(out);
      }
    });

    ws.addEventListener("error", (event) => {
      const detail = event && (event.message || (event.error && event.error.message));
      fail(`Soniox WebSocket error on chunk ${chunkIndex + 1}/${totalChunks}${detail ? ": " + detail : ""}`);
    });
    ws.addEventListener("close", (event) => {
      if (!settled) {
        fail(`Soniox WebSocket closed early on chunk ${chunkIndex + 1}/${totalChunks} code=${event.code || "unknown"}`);
      }
    });
  });
}

async function synthesizeTTSChunk(text, speed, callId, chunkIndex, totalChunks) {
  const maxAttempts = callId === "warmup" ? 2 : 3;
  let lastErr = null;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await synthesizeTTSChunkOnce(text, speed, callId, chunkIndex, totalChunks, attempt);
    } catch (err) {
      lastErr = err;
      if (attempt >= maxAttempts) break;
      const backoff = attempt === 1 ? 250 : 700;
      console.log(`[tts-retry] [call ${callId}] chunk ${chunkIndex + 1}/${totalChunks} attempt ${attempt} failed — retrying in ${backoff}ms: ${err.message}`);
      await sleep(backoff);
    }
  }
  throw lastErr || new Error("Soniox TTS failed");
}

async function synthesizeUpliftTTS(text, callId, attempt = 1) {
  if (!UPLIFT_API_KEY) throw new Error("UPLIFT_API_KEY not set");
  const started = Date.now();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Number(process.env.UPLIFT_TTS_TIMEOUT_MS || 20000));
  try {
    const response = await fetch(UPLIFT_TTS_URL, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${UPLIFT_API_KEY}`,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
      },
      body: JSON.stringify({
        voiceId: UPLIFT_TTS_VOICE,
        text: String(text || "").trim(),
        outputFormat: UPLIFT_TTS_FORMAT
      }),
      signal: controller.signal
    });
    if (!response.ok) {
      let detail = "";
      try { detail = (await response.text()).slice(0, 400); } catch (e) {}
      throw new Error(`Uplift TTS HTTP ${response.status}${detail ? ": " + detail : ""}`);
    }
    const arrayBuffer = await response.arrayBuffer();
    const audio = Buffer.from(arrayBuffer);
    if (!audio.length) throw new Error("Uplift returned empty audio");
    console.log(`[timing] [call ${callId}] UPLIFT TTS complete: +${Date.now() - started}ms (${audio.length} bytes, attempt ${attempt}, voice=${UPLIFT_TTS_VOICE})`);
    return audio;
  } catch (err) {
    if (err && err.name === "AbortError") throw new Error("Uplift TTS timed out");
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

async function synthesizeUpliftWithRetry(text, callId) {
  let lastErr;
  for (let attempt = 1; attempt <= 2; attempt++) {
    try { return await synthesizeUpliftTTS(text, callId, attempt); }
    catch (err) {
      lastErr = err;
      if (attempt < 2) {
        console.log(`[uplift-retry] [call ${callId}] attempt ${attempt} failed — retrying in 300ms: ${err.message}`);
        await sleep(300);
      }
    }
  }
  throw lastErr || new Error("Uplift TTS failed");
}

async function synthesizePrimaryTTS(text, tone, callId) {
  if (TTS_PROVIDER === "uplift") {
    try {
      const audio = await synthesizeUpliftWithRetry(text, callId);
      return { audio, provider:"UPLIFT", voice:UPLIFT_TTS_VOICE };
    } catch (upliftErr) {
      if (!ENABLE_SONIOX_TTS_FALLBACK || !SONIOX_API_KEY) throw upliftErr;
      console.log(`[tts-fallback] [call ${callId}] Uplift failed; using Soniox fallback: ${upliftErr.message}`);
      const speed = tone === "empathetic" ? 0.88 : 1.0;
      const audio = await synthesizeTTSChunk(text, speed, callId, 0, 1);
      return { audio, provider:"SONIOX_FALLBACK", voice:SONIOX_TTS_VOICE };
    }
  }
  const speed = tone === "empathetic" ? 0.88 : 1.0;
  const audio = await synthesizeTTSChunk(text, speed, callId, 0, 1);
  return { audio, provider:"SONIOX", voice:SONIOX_TTS_VOICE };
}

async function handleTTSRequest(text, tone, callId, res) {
  const primaryAvailable = TTS_PROVIDER === "uplift" ? !!UPLIFT_API_KEY : !!SONIOX_API_KEY;
  const fallbackAvailable = ENABLE_SONIOX_TTS_FALLBACK && !!SONIOX_API_KEY;
  if (!primaryAvailable && !fallbackAvailable) {
    return res.status(501).json({ error: { message: "No server TTS key configured — set UPLIFT_API_KEY (recommended) or SONIOX_API_KEY." } });
  }
  if (!text || typeof text !== "string") {
    return res.status(400).json({ error: { message: "Request must include text." } });
  }
  if (!callId) {
    return res.status(400).json({ error: { message: "Request must include callId." } });
  }

  const started = Date.now();
  const cacheFile = ttsCachePath(text, tone);
  if (fs.existsSync(cacheFile)) {
    const audio = fs.readFileSync(cacheFile);
    let meta = null;
    try { meta = JSON.parse(fs.readFileSync(ttsCacheMetaPath(text, tone), "utf8")); } catch (e) {}
    const provider = (meta && meta.provider) || (TTS_PROVIDER === "uplift" ? "UPLIFT" : "SONIOX");
    const voice = (meta && meta.voice) || (provider.startsWith("UPLIFT") ? UPLIFT_TTS_VOICE : SONIOX_TTS_VOICE);
    console.log(`[tts-cache] [call ${callId}] HIT ${text.length} chars (${audio.length} bytes)`);
    logVoiceSource(callId, provider, voice, true);
    res.writeHead(200, {
      "Content-Type": "audio/mpeg",
      "Content-Length": String(audio.length),
      "Cache-Control": "public, max-age=31536000, immutable"
    });
    return res.end(audio);
  }

  console.log(`[tts-cache] [call ${callId}] MISS — provider=${TTS_PROVIDER.toUpperCase()}, voice=${TTS_PROVIDER === "uplift" ? UPLIFT_TTS_VOICE : SONIOX_TTS_VOICE}, ${text.length} chars total`);

  activeLiveTTS++;
  try {
    // Exactly ONE synthesis request per Sania turn. This prevents the old
    // multi-phase talk/listen/talk effect. Uplift is primary; Soniox TTS is
    // fallback only when explicitly enabled and Uplift is unavailable.
    const result = await synthesizePrimaryTTS(text, tone, callId);
    const audio = result.audio;
    logVoiceSource(callId, result.provider, result.voice, false);
    console.log(`[timing] [call ${callId}] buffered TTS ready: +${Date.now() - started}ms (${audio.length} bytes, provider=${result.provider})`);
    try {
      fs.writeFileSync(cacheFile, audio);
      fs.writeFileSync(ttsCacheMetaPath(text, tone), JSON.stringify({provider:result.provider,voice:result.voice}, null, 2));
    } catch (e) {}
    res.writeHead(200, {
      "Content-Type": "audio/mpeg",
      "Content-Length": String(audio.length),
      "Cache-Control": "public, max-age=31536000, immutable"
    });
    res.end(audio);
  } catch (err) {
    console.log(`🚩 TTS FAILED [call ${callId}] after retries: ${err.message}`);
    if (!res.headersSent) {
      res.status(502).json({ error: { message: err.message } });
    } else {
      try { res.end(); } catch (e) {}
    }
  } finally {
    activeLiveTTS = Math.max(0, activeLiveTTS - 1);
  }
}

// at this URL, so the BROWSER's own native progressive-streaming playback
// kicks in (the same mechanism internet radio streams use) instead of our
// code manually buffering a full blob before playback can start.
app.get("/api/tts", (req, res) => {
  handleTTSRequest(req.query.text, req.query.tone, req.query.callId, res);
});

// POST variant kept for compatibility / non-browser callers.
app.post("/api/tts", (req, res) => {
  const { text, tone, callId } = req.body || {};
  handleTTSRequest(text, tone, callId, res);
});

// ==========================================================================
// Speech-to-text: bridges the customer's mic audio to Soniox's real-time STT
// WebSocket. This replaces the browser's own SpeechRecognition — the reason
// is Soniox's "enable_endpoint_detection" gives us a REAL signal for "the
// customer stopped talking," instead of our old homemade JavaScript silence
// timer guessing at it. Protocol confirmed from Soniox's own reference
// implementation (same source as the TTS fix earlier), not guessed.
//
// One STT connection per phone call (not per utterance) — it stays open for
// the whole call, and Soniox's endpoint detection tells us turn-by-turn when
// the customer is done talking.
// ==========================================================================
const http = require("http");

// stt-rt-v5 specifically — required for endpoint_sensitivity tuning below,
// which earlier STT models reject outright.
const SONIOX_STT_MODEL = process.env.SONIOX_STT_MODEL || "stt-rt-v5";
// Overridable so this can be pointed at a local mock server for testing —
// production always uses the real Soniox endpoint by default.
const SONIOX_STT_URL = process.env.SONIOX_STT_URL || "wss://stt-rt.soniox.com/transcribe-websocket";

const httpServer = http.createServer(app);
const wss = new WebSocketServer({ server: httpServer, path: "/ws/stt" });

wss.on("connection", (clientWs, req) => {
  let callId = "unknown";
  try {
    const url = new URL(req.url, `http://${req.headers.host}`);
    callId = url.searchParams.get("callId") || "unknown";
  } catch (e) { /* keep default */ }

  liveCallActive = true;
  console.log(`[stt] [call ${callId}] client connected`);

  if (!SONIOX_API_KEY) {
    try { clientWs.send(JSON.stringify({ error: "SONIOX_API_KEY not set on server" })); } catch (e) {}
    clientWs.close();
    return;
  }

  let sonioxReady = false;
  let closed = false;
  const audioQueue = [];
  const MAX_QUEUE = 200; // safety cap — drop oldest audio rather than grow unbounded if Soniox is slow to connect

  const sonioxWs = new WebSocket(SONIOX_STT_URL);

  sonioxWs.addEventListener("open", () => {
    sonioxWs.send(JSON.stringify({
      api_key: SONIOX_API_KEY,
      model: SONIOX_STT_MODEL,
      enable_endpoint_detection: true,
      enable_non_final_tokens: true,
      language_hints: ["ur"],
      audio_format: "pcm_s16le",
      sample_rate: 16000,
      num_channels: 1,
      // Negative sensitivity = less eager to declare "they're done talking,"
      // which is what tolerates a natural pause mid-sentence instead of
      // cutting the customer off before a long sentence is finished.
      // Range is -1.0 to 1.0; tune via env var if this needs adjusting.
      endpoint_sensitivity: parseFloat(process.env.SONIOX_ENDPOINT_SENSITIVITY || "0.20"),
      // Upper bound on how long it'll wait after speech stops before
      // finalizing — pushed toward the documented max (3000) for the same
      // reason: more tolerance for a mid-sentence breath.
      max_endpoint_delay_ms: parseInt(process.env.SONIOX_MAX_ENDPOINT_DELAY_MS || "650", 10)
    }));
    sonioxReady = true;
    while (audioQueue.length) {
      sonioxWs.send(audioQueue.shift());
    }
  });

  sonioxWs.addEventListener("message", (event) => {
    // Relay Soniox's transcription/endpoint messages straight through to
    // the browser — the client decides what to do with tokens and <end>.
    if (clientWs.readyState === 1) {
      try { clientWs.send(event.data); } catch (e) {}
    }
  });

  const connectionStartTime = Date.now();

  sonioxWs.addEventListener("error", (event) => {
    console.log(`[stt] [call ${callId}] Soniox STT connection ERROR after ${Date.now() - connectionStartTime}ms — ${event.message || event.error || "no details available"}`);
    if (!closed) { closed = true; try { clientWs.close(); } catch (e) {} }
  });

  sonioxWs.addEventListener("close", (event) => {
    console.log(`[stt] [call ${callId}] Soniox STT connection CLOSED after ${Date.now() - connectionStartTime}ms — code=${event.code} reason="${event.reason || "(none given)"}" wasClean=${event.wasClean}`);
    if (!closed) { closed = true; try { clientWs.close(); } catch (e) {} }
  });

  const keepaliveInterval = setInterval(() => {
    if (sonioxWs.readyState === 1) {
      try { sonioxWs.send(JSON.stringify({ type: "keepalive" })); } catch (e) {}
    }
  }, 5000);

  // Browser -> server heartbeat. This is application-level on purpose:
  // it keeps intermediaries/proxies from considering an otherwise quiet
  // WebSocket idle while Sania is speaking and mic audio is intentionally
  // paused. Text control frames are handled locally and NEVER forwarded to
  // Soniox as PCM audio.
  clientWs.on("message", (data, isBinary) => {
    if (!isBinary) {
      let control = null;
      try { control = JSON.parse(data.toString()); } catch (e) {}
      if (control && control.type === "client_keepalive") {
        try {
          clientWs.send(JSON.stringify({ type: "server_keepalive", ts: Date.now() }));
        } catch (e) {}
        return;
      }
      return;
    }

    // Binary PCM audio frames from the browser — forward to Soniox, or
    // queue briefly if Soniox's own connection isn't ready yet.
    if (sonioxReady && sonioxWs.readyState === 1) {
      try { sonioxWs.send(data); } catch (e) {}
    } else {
      audioQueue.push(data);
      if (audioQueue.length > MAX_QUEUE) audioQueue.shift();
    }
  });

  clientWs.on("close", (code, reason) => {
    liveCallActive = false;
    clearInterval(keepaliveInterval);
    console.log(`[stt] [call ${callId}] client disconnected after ${Date.now() - connectionStartTime}ms — code=${code} reason="${reason ? reason.toString() : "(none given)"}"`);
    try { sonioxWs.close(); } catch (e) {}
  });

  clientWs.on("error", (err) => {
    clearInterval(keepaliveInterval);
    console.log(`[stt] [call ${callId}] client WebSocket ERROR after ${Date.now() - connectionStartTime}ms — ${err.message}`);
    try { sonioxWs.close(); } catch (e) {}
  });
});

const PREWARM_TTS_LINES = [
  "السلام علیکم، سیئرل فارمیسی سے ثانیہ بات کر رہی ہوں؛ آپ کا نام کیا ہے؟",
  "وعلیکم السلام، جی ضرور؛ آپ اپنا نام بتا دیجیے۔",
  "جی، نام ذرا ایک بار پھر بتا دیجیے۔",
  "شکریہ، ایکسٹور کے بارے میں آپ کیا جاننا چاہیں گے؟",
  "معذرت، میں صرف ایکسٹور دوا سے متعلق سوالات میں آپ کی مدد کر سکتی ہوں؛ ایکسٹور کے بارے میں کیا جاننا چاہیں گے؟",
  "جی، کیا آپ کے پاس ڈاکٹر کا ایکسٹور کا نسخہ موجود ہے؟",
  "ٹھیک ہے، براہِ کرم اپنا ایکسٹور کا نسخہ کیمرے کے سامنے دکھا دیجیے۔",
  "جی، نسخہ واضح ہے؛ آپ کا بلڈ پریشر عموماً کتنا رہتا ہے؟",
  "ٹھیک ہے، شکریہ؛ اب ایکسٹور کے بارے میں آپ کیا جاننا چاہیں گے؟",
  "ایکسٹور نسخے والی دوا ہے؛ آپ کس علاقے میں ہیں تاکہ میں قریب ڈاکٹر بتا سکوں؟",
  "آپ کس علاقے میں ہیں تاکہ میں قریب کی فارمیسی بتا سکوں؟",
  "ایکسٹور ہائی بلڈ پریشر کی دوا ہے، جس میں ایملوڈیپین اور والسارٹن شامل ہیں۔",
  "ایکسٹور 5/80، 5/160 اور 10/160 ملی گرام میں دستیاب ہے؛ درست طاقت ڈاکٹر طے کرتا ہے۔",
  "عام سائیڈ ایفیکٹس میں چکر، سر درد، کمزوری اور ٹخنوں کی سوجن شامل ہو سکتی ہے۔",
  "اگر کوئی سائیڈ ایفیکٹ ہو رہا ہے تو علامت بتا دیجیے؛ میں اسی کے مطابق مختصر رہنمائی دوں گی۔",
  "ہلکے سر درد میں آرام اور پانی مفید ہو سکتا ہے؛ پیراسیٹامول بعض بالغ افراد میں مناسب ہوتا ہے، مگر جگر کی بیماری یا الرجی ہو تو پہلے ڈاکٹر یا فارماسسٹ سے پوچھیں۔",
  "چکر ہوں تو بیٹھ یا لیٹ جائیں، پانی پئیں اور بلڈ پریشر چیک کریں؛ بےہوشی، سینے کے درد یا شدید کمزوری میں فوری طبی مدد لیں۔",
  "ٹخنوں یا پاؤں کی سوجن میں خود سے پیشاب آور دوا شروع نہ کریں؛ اپنے ڈاکٹر سے ایکسٹور کی خوراک یا علاج کا جائزہ کروائیں۔",
  "کمزوری ہو تو بلڈ پریشر چیک کریں، پانی پئیں اور آرام کریں؛ بہت کم بلڈ پریشر، بےہوشی یا شدید کمزوری میں فوراً ڈاکٹر سے رابطہ کریں۔",
  "متلی میں پانی اور ہلکی غذا لیں؛ متلی کی دوا خود سے شامل کرنے کے بجائے ڈاکٹر یا فارماسسٹ سے مناسب دوا کنفرم کریں۔",
  "دھڑکن تیز ہو تو آرام کریں اور بلڈ پریشر چیک کریں؛ سینے کے درد، سانس کی دشواری یا بےہوشی کے ساتھ ہو تو فوری طبی مدد لیں۔",
  "جی، کچھ ہلکے سائیڈ ایفیکٹس کے لیے دوسری دوا استعمال کی جا سکتی ہے؛ مثلاً ہلکے سر درد میں پیراسیٹامول بعض بالغ افراد کے لیے مناسب ہو سکتا ہے۔ لیکن چکر، سوجن، متلی یا دوسرے سائیڈ ایفیکٹس کے لیے کون سی دوا مناسب ہے، یہ ڈاکٹر یا فارماسسٹ سے کنفرم کرنا ضروری ہے، اس لیے ایکسٹور کے ساتھ خود سے نئی دوا شروع نہ کریں۔",
  "ایکسٹور کے ساتھ دوسری دوائیں لی جا سکتی ہیں، لیکن یہ دوا پر منحصر ہے؛ کوئی نئی دوا، خاص طور پر آئبوپروفین، ڈائیکلوفیناک، پوٹاشیم سپلیمنٹ یا پیشاب آور دوا، ڈاکٹر یا فارماسسٹ سے کنفرم کیے بغیر شروع نہ کریں۔",
  "جی، اگر آپ علامت بتا دیں تو میں عمومی طور پر بتا سکتی ہوں کیا کیا جا سکتا ہے، لیکن نئی دوا شروع کرنے سے پہلے ڈاکٹر یا فارماسسٹ سے کنفرم کرنا ضروری ہے۔",
  "خوراک بھول جائیں تو یاد آنے پر لیں، مگر اگلی خوراک قریب ہو تو دگنی خوراک نہ لیں۔",
  "ایکسٹور ڈاکٹر کے نسخے کے مطابق روزانہ تقریباً ایک ہی وقت پر، کھانے کے ساتھ یا بغیر لی جا سکتی ہے۔",
  "ایکسٹور روزانہ تقریباً ایک ہی وقت پر لیں؛ صبح یا شام وہی وقت رکھیں جو ڈاکٹر نے بتایا ہو۔",
  "ایکسٹور کھانے کے ساتھ یا بغیر کھانے کے لی جا سکتی ہے؛ اسے پانی کے ساتھ نگلیں۔",
  "حمل میں ایکسٹور استعمال نہیں کرنی چاہیے؛ فوراً اپنے ڈاکٹر سے مشورہ کریں۔",
  "دودھ پلانے کے دوران ایکسٹور کے بارے میں اپنے ڈاکٹر سے مشورہ کریں؛ خود سے استعمال شروع نہ کریں۔",
  "گردے یا جگر کے مسئلے میں ایکسٹور ڈاکٹر کی نگرانی میں لینی چاہیے؛ اپنی خوراک خود تبدیل نہ کریں۔",
  "پوٹاشیم سپلیمنٹ یا پوٹاشیم بڑھانے والی دوا کے ساتھ ایکسٹور لینے سے پہلے ڈاکٹر سے مشورہ کریں۔",
  "ایکسٹور عام طور پر 18 سال سے کم عمر افراد کے لیے تجویز نہیں کی جاتی؛ ڈاکٹر سے مشورہ کریں۔"
];

async function warmCommonTTS(){
  // Disabled by default so starting the app never burns Uplift credits or
  // competes with a live caller. Set ENABLE_TTS_PREWARM=1 if desired.
  if (process.env.ENABLE_TTS_PREWARM !== "1" || prewarmRunning) return;
  if (TTS_PROVIDER === "uplift" && !UPLIFT_API_KEY) return;
  if (TTS_PROVIDER !== "uplift" && !SONIOX_API_KEY) return;
  prewarmRunning = true;
  console.log(`[tts-cache] prewarming ${PREWARM_TTS_LINES.length} common Extor lines gently in background...`);
  try {
    for (let i=0; i<PREWARM_TTS_LINES.length; i++){
      if (liveCallActive) {
        console.log(`[tts-cache] prewarm paused for live call at ${i}/${PREWARM_TTS_LINES.length}`);
        break;
      }
      const line=PREWARM_TTS_LINES[i];
      const cacheFile=ttsCachePath(line,"normal");
      if (fs.existsSync(cacheFile)) continue;

      // Real patient speech always wins. Do not compete with live TTS sockets.
      while (activeLiveTTS > 0 && !liveCallActive) await sleep(250);
      await sleep(350); // gentle pacing avoids hammering Soniox on startup

      try{
        while (activeLiveTTS > 0 && !liveCallActive) await sleep(250);
        const audio = await synthesizePrimaryTTS(line, "normal", "warmup");
        fs.writeFileSync(cacheFile,audio);
        console.log(`[tts-cache] warmed ${i+1}/${PREWARM_TTS_LINES.length}`);
      }catch(e){
        // A failed warmup is never fatal; skip this line and continue later.
        console.log(`[tts-cache] warmup line ${i+1} skipped after retries: ${e.message}`);
        await sleep(1200);
      }
    }
  } finally {
    prewarmRunning = false;
  }
}

const PORT = process.env.PORT || 3000;
httpServer.listen(PORT, () => {
  console.log(`Pharmacy voice agent server listening on port ${PORT}`);
  console.log(`[voice] STT=SONIOX | TTS=${TTS_PROVIDER.toUpperCase()}${TTS_PROVIDER === "uplift" ? ` | voice=${UPLIFT_TTS_VOICE}` : ` | voice=${SONIOX_TTS_VOICE}`}`);
  console.log(`[voice] Uplift key=${UPLIFT_API_KEY ? "READY" : "MISSING"} | Soniox key=${SONIOX_API_KEY ? "READY" : "MISSING"} | Claude key=${ANTHROPIC_API_KEY ? "READY" : "MISSING"}`);
  setTimeout(()=>warmCommonTTS().catch(()=>{}), 2500);
});
