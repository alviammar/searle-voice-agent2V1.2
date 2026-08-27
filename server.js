const express = require("express");
const path = require("path");
const fs = require("fs");
const crypto = require("crypto");
const { WebSocket, WebSocketServer } = require("ws");
const { Pool } = require("pg");

const app = express();
// The UI may be hosted separately (for example on Netlify) while this
// always-on Node/WebSocket backend runs on Render/Railway/Fly/Cloud Run.
// No cookies or browser credentials are used, so a simple CORS policy is
// appropriate for the demo. Lock this to your production domain later.
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", process.env.CORS_ORIGIN || "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, x-training-key");
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


// Persistent learning store. On Render, set DATABASE_URL to a PostgreSQL
// database. If DATABASE_URL is absent (for local Mac testing), the agent
// safely falls back to the legacy JSON files in /data.
const DATABASE_URL = process.env.DATABASE_URL || "";
const TRAINING_IMPORT_KEY = process.env.TRAINING_IMPORT_KEY || "";
const SUPERVISED_TRAINING_FILE = path.join(__dirname, "data", "sania_supervised_training_v1.json");
const learningPool = DATABASE_URL ? new Pool({
  connectionString: DATABASE_URL,
  ssl: process.env.PGSSLMODE === "disable" ? false : { rejectUnauthorized: false },
  max: 5,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 10000
}) : null;
let learningDbReady = false;
let learningDbInitError = null;

async function initLearningDb(){
  if (!learningPool) return false;
  try {
    await learningPool.query(`
      CREATE TABLE IF NOT EXISTS sania_learned_answers (
        id BIGSERIAL PRIMARY KEY,
        intent TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        hits INTEGER NOT NULL DEFAULT 0,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE(intent, question)
      );
      CREATE INDEX IF NOT EXISTS idx_sania_answers_intent ON sania_learned_answers(intent);

      CREATE TABLE IF NOT EXISTS sania_learned_phrases (
        id BIGSERIAL PRIMARY KEY,
        intent TEXT NOT NULL,
        phrase TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE(intent, phrase)
      );
      CREATE INDEX IF NOT EXISTS idx_sania_phrases_intent ON sania_learned_phrases(intent);

      CREATE TABLE IF NOT EXISTS sania_learning_events (
        id BIGSERIAL PRIMARY KEY,
        ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        call_id TEXT,
        text TEXT,
        intent TEXT,
        matched BOOLEAN NOT NULL DEFAULT FALSE,
        used_claude BOOLEAN NOT NULL DEFAULT FALSE,
        latency_ms INTEGER,
        source TEXT
      );
      CREATE INDEX IF NOT EXISTS idx_sania_events_ts ON sania_learning_events(ts DESC);

      CREATE TABLE IF NOT EXISTS sania_training_packs (
        version TEXT PRIMARY KEY,
        checksum TEXT,
        records_total INTEGER NOT NULL DEFAULT 0,
        answers_imported INTEGER NOT NULL DEFAULT 0,
        phrases_imported INTEGER NOT NULL DEFAULT 0,
        imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
      );
    `);
    learningDbReady = true;
    console.log('[learning-db] PostgreSQL READY — learning is persistent/global');
    return true;
  } catch (err) {
    learningDbInitError = err;
    learningDbReady = false;
    console.error('[learning-db] PostgreSQL unavailable; using JSON fallback:', err.message);
    return false;
  }
}
const learningDbReadyPromise = initLearningDb().then(async ready => { if(ready){ setTimeout(()=>reclassifyExistingOtherExtor().catch(()=>{}), 800); } return ready; });

async function dbIsReady(){
  if (!learningPool) return false;
  if (learningDbReady) return true;
  await learningDbReadyPromise.catch(()=>false);
  return learningDbReady;
}

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

const CANONICAL_LEARN_INTENTS = new Set([
  "what_is","strengths","dose_frequency","side_effects","side_effect_help","side_effect_medicine",
  "other_medicines","missed_dose","how_to_take","timing","food","pregnancy","breastfeeding",
  "kidney_liver","potassium","under18","stopping","co_extor","overdose","emergency","price",
  "competitor","purchase","precautions","bp_related","alcohol","monitoring","effectiveness","onset","storage","refill","travel","other_extor"
]);

function normalizeIntentName(value){
  const x=String(value||"").trim().toLowerCase();
  const aliases={
    unknown_extor:"other_extor",
    kidney_precautions:"kidney_liver", renal_precautions:"kidney_liver", liver_precautions:"kidney_liver",
    interaction:"other_medicines", interactions:"other_medicines", medicine_interaction:"other_medicines", drug_interaction:"other_medicines",
    dose:"dose_frequency", dosage_frequency:"dose_frequency", competitors:"competitor", alternatives:"competitor", cost:"price",
    safety:"precautions", caution:"precautions", precaution:"precautions"
  };
  return aliases[x]||x;
}
function learningText(value){ return String(value||"").toLowerCase().replace(/[؟?۔.,!،;؛:()"']/g," ").replace(/\s+/g," ").trim(); }
function hasLearnAny(t,list){ return list.some(x=>t.includes(x)); }
function looksLikeChitChatForLearning(text){
  const t=learningText(text);
  if(!t || t.length<5) return true;
  const chit=[
    "thank you","thanks","thankyou","shukriya","شکریہ","theek hai","ٹھیک ہے","okay","ok ","acha theek","اچھا ٹھیک",
    "allah hafiz","اللہ حافظ","good bye","goodbye","bye","چلیں ٹھیک","chalein theek","walikum salam","وعلیکم السلام","assalam o alaikum","السلام علیکم"
  ];
  const medical=hasLearnAny(t,["extor","ایکسٹور","kidney","renal","گرد","liver","جگر","side effect","سائیڈ","dose","خوراک","tablet","گولی","medicine","دوا","bp","blood pressure","بلڈ پریشر","pregnan","حمل","price","قیمت","competitor","alternative","متبادل","5/80","5/160","10/160"]);
  return !medical && hasLearnAny(t,chit);
}
function inferLearningIntent(text, proposedIntent){
  const t=learningText(text); const proposed=normalizeIntentName(proposedIntent);
  // Strong topic-first rules. These always outrank a generic proposed intent.
  if(hasLearnAny(t,["kidney","kidneys","renal","گردے","گردہ","گردوں","gurday","gurda"])) return "kidney_liver";
  if(hasLearnAny(t,["liver","hepatic","جگر","jigar"])) return "kidney_liver";
  if(hasLearnAny(t,["pregnant","pregnancy","حمل","حاملہ","hamal"])) return "pregnancy";
  if(hasLearnAny(t,["breastfeed","breastfeeding","دودھ پلا","feeding baby","nursing"])) return "breastfeeding";
  if(hasLearnAny(t,["potassium","پوٹاشیم"])) return "potassium";
  if(hasLearnAny(t,["under 18","under18","18 years","18 سال","child","children","teen","بچہ","بچوں"])) return "under18";
  if(hasLearnAny(t,["twice","twice a day","two times","2 times","2 tablets","two tablets","double dose","do baar","2 baar","دو بار","دو دفعہ","دو گولی","دو گولیاں","دن میں دو","ایک دن میں دو"])) return "dose_frequency";
  if(hasLearnAny(t,["price","cost","rate","قیمت","دام","روپے","pkr","kitne ki","kitni ki","کتنے کی","کتنی کی"])) return "price";
  if(hasLearnAny(t,["competitor","competition","alternative","alternatives","substitute","substitutes","compare","comparison","versus","better","exforge","avsar","amlortan","amstan","dioplus","newday","valam","valtec","متبادل","مقابلہ"])) return "competitor";
  if(hasLearnAny(t,["missed dose","miss dose","dose miss","bhool","بھول"])) return "missed_dose";
  if(hasLearnAny(t,["panadol","paracetamol","ibuprofen","brufen","other medicine","another medicine","koi aur dawa","ساتھ کون سی دوا","interaction","ساتھ لے","ساتھ لوں"])) return "other_medicines";
  const symptom=hasLearnAny(t,["chakkar","dizziness","dizzy","swelling","soojan","سوجن","headache","سر درد","weakness","کمزوری","nausea","متلی","palpitation","دھڑکن"]);
  if(symptom && hasLearnAny(t,["mujhe","مجھے","having","feel","feeling","ho raha","ہو رہا","what should","kya kar","کیا کروں"])) return "side_effect_help";
  if(hasLearnAny(t,["side effect","side effects","سائیڈ ایفیکٹ","نقصان"])) return "side_effects";
  if(hasLearnAny(t,["alcohol","drink alcohol","شراب","sharab"])) return "alcohol";
  if(hasLearnAny(t,["monitor","monitoring","check bp","bp check","blood pressure check","کتنی بار چیک","مانیٹر"])) return "monitoring";
  if(hasLearnAny(t,["effective","effectiveness","working","work kar","اثر کر","کام کر","bp control"])) return "effectiveness";
  if(hasLearnAny(t,["how long to work","how quickly","start working","effect start","اثر کب","کتنی دیر میں اثر","onset"])) return "onset";
  if(hasLearnAny(t,["store","storage","keep medicine","room temperature","fridge","refrigerator","محفوظ رکھ","کہاں رکھ"])) return "storage";
  if(hasLearnAny(t,["refill","repeat prescription","next pack","دوبارہ نسخہ","ریفل"])) return "refill";
  if(hasLearnAny(t,["travel","travelling","flight","airport","سفر","جہاز"])) return "travel";
  if(hasLearnAny(t,["stop extor","stop taking","band kar","بند کر","چھوڑ دوں","چھوڑ سکتا","چھوڑ سکتی"])) return "stopping";
  if(hasLearnAny(t,["co extor","co-extor","کو ایکسٹور"])) return "co_extor";
  if(hasLearnAny(t,["overdose","too many tablets","extra tablets","زیادہ گولیاں","زیادہ خوراک"])) return "overdose";
  if(hasLearnAny(t,["chest pain","سینے میں درد","difficulty breathing","سانس","faint","بے ہوش"])) return "emergency";
  if(hasLearnAny(t,["5/80","5 80","5/160","5 160","10/160","10 160","strength","power","variation","variant","طاقت","ورائٹی"])) return "strengths";
  if(hasLearnAny(t,["with food","without food","empty stomach","khane","کھانے","خالی پیٹ"])) return "food";
  if(hasLearnAny(t,["what time","when should","kab","کس وقت","کب لوں","morning","evening","night","صبح","شام","رات"])) return "timing";
  if(hasLearnAny(t,["how to take","how should i take","kaise loon","کیسے لوں","water","pani","پانی","milk","doodh","دودھ"])) return "how_to_take";
  if(hasLearnAny(t,["where can i buy","where to buy","pharmacy","medical store","کہاں ملے","فارمیسی","خرید"])) return "purchase";
  if(hasLearnAny(t,["bp","blood pressure","بلڈ پریشر","پریشر"]) && hasLearnAny(t,["high","low","normal","زیادہ","کم","نارمل"])) return "bp_related";
  if(hasLearnAny(t,["احتیاط","ehtiyat","precaution","precautions","caution","safe to use","is it safe","کیا محفوظ","محفوظ ہے"])) return "precautions";
  if(hasLearnAny(t,["what is extor","extor kya hai","ایکسٹور کیا ہے","used for","kis liye","کس لیے"])) return "what_is";
  // A specific Claude intent is allowed if it is in the canonical list. Generic
  // other_extor never overrides the full-question rules above.
  if(proposed && proposed!=="other_extor" && CANONICAL_LEARN_INTENTS.has(proposed)) return proposed;
  return "other_extor";
}

async function reclassifyExistingOtherExtor(){
  if(!(await dbIsReady())) return {ok:false, reason:"db_not_ready"};
  const stats={answersMoved:0,answersRemoved:0,phrasesMoved:0,phrasesRemoved:0,eventsMoved:0};
  try{
    const answers=(await learningPool.query(`SELECT id,question,answer,hits FROM sania_learned_answers WHERE intent IN ('other_extor','unknown_extor')`)).rows;
    for(const row of answers){
      if(looksLikeChitChatForLearning(row.question)){
        await learningPool.query(`DELETE FROM sania_learned_answers WHERE id=$1`,[row.id]); stats.answersRemoved++; continue;
      }
      const ni=inferLearningIntent(row.question,null);
      if(ni && ni!=="other_extor"){
        await learningPool.query(`INSERT INTO sania_learned_answers(intent,question,answer,hits,updated_at) VALUES($1,$2,$3,$4,NOW()) ON CONFLICT(intent,question) DO UPDATE SET answer=EXCLUDED.answer,hits=GREATEST(sania_learned_answers.hits,EXCLUDED.hits),updated_at=NOW()`,[ni,row.question,row.answer,row.hits||0]);
        await learningPool.query(`DELETE FROM sania_learned_answers WHERE id=$1`,[row.id]); stats.answersMoved++;
      }
    }
    const phrases=(await learningPool.query(`SELECT id,phrase FROM sania_learned_phrases WHERE intent IN ('other_extor','unknown_extor')`)).rows;
    for(const row of phrases){
      if(looksLikeChitChatForLearning(row.phrase)){
        await learningPool.query(`DELETE FROM sania_learned_phrases WHERE id=$1`,[row.id]); stats.phrasesRemoved++; continue;
      }
      const ni=inferLearningIntent(row.phrase,null);
      if(ni && ni!=="other_extor"){
        await learningPool.query(`INSERT INTO sania_learned_phrases(intent,phrase) VALUES($1,$2) ON CONFLICT(intent,phrase) DO NOTHING`,[ni,row.phrase]);
        await learningPool.query(`DELETE FROM sania_learned_phrases WHERE id=$1`,[row.id]); stats.phrasesMoved++;
      }
    }
    const events=(await learningPool.query(`SELECT id,text FROM sania_learning_events WHERE intent IN ('other_extor','unknown_extor')`)).rows;
    for(const row of events){ const ni=inferLearningIntent(row.text,null); if(ni && ni!=="other_extor"){ await learningPool.query(`UPDATE sania_learning_events SET intent=$1 WHERE id=$2`,[ni,row.id]); stats.eventsMoved++; } }
    console.log(`[learning-migrate] ${JSON.stringify(stats)}`);
    return {ok:true,...stats};
  }catch(e){ console.error('[learning-migrate] failed:',e.message); return {ok:false,error:e.message}; }
}
function isDuplicateLearningEvent(callId,text,intent,source){
  const key=[String(callId||""),learningText(text),String(intent||""),String(source||"")].join("|"); const now=Date.now(),prev=recentLearningEvents.get(key)||0; recentLearningEvents.set(key,now);
  if(recentLearningEvents.size>500){ for(const [k,ts] of recentLearningEvents) if(now-ts>60000) recentLearningEvents.delete(k); }
  return !!prev && (now-prev)<LEARN_DEDUPE_WINDOW_MS;
}
async function addLearnedPhrase(intent,text){
  intent=normalizeIntentName(intent); const normalized=learningText(text); if(!intent||normalized.length<3)return;
  if (await dbIsReady()) {
    try {
      await learningPool.query(
        `INSERT INTO sania_learned_phrases(intent, phrase) VALUES($1,$2)
         ON CONFLICT(intent, phrase) DO NOTHING`,
        [intent, normalized]
      );
      return;
    } catch (e) { console.error('[learning-db] phrase save failed:', e.message); }
  }
  try{ const learned=JSON.parse(fs.readFileSync(LEARNED_PHRASES_FILE,"utf8")||"{}"); learned[intent]=Array.isArray(learned[intent])?learned[intent]:[]; if(!learned[intent].includes(normalized)){ learned[intent].push(normalized); learned[intent]=learned[intent].slice(-250); fs.writeFileSync(LEARNED_PHRASES_FILE,JSON.stringify(learned,null,2)); } }catch(e){}
}

app.post("/api/learn", async (req, res) => {
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
  if (!duplicate) {
    if (await dbIsReady()) {
      try {
        await learningPool.query(
          `INSERT INTO sania_learning_events(ts,call_id,text,intent,matched,used_claude,latency_ms,source)
           VALUES($1,$2,$3,$4,$5,$6,$7,$8)`,
          [safe.ts,safe.callId,safe.text,safe.intent,safe.matched,safe.usedClaude,safe.latencyMs,safe.source]
        );
      } catch (e) { console.error('[learning-db] event save failed:', e.message); }
    } else {
      try { fs.appendFileSync(LEARNING_LOG, JSON.stringify(safe) + "\n"); } catch (e) {}
    }
  } else console.log(`[learning-dedupe] skipped duplicate intent=${safe.intent} source=${safe.source}`);

  if (safe.source !== "CLAUDE_REQUEST") {
    const label = safe.source === "CLAUDE_RESPONSE" ? "CLAUDE" : safe.source;
    console.log("\n============================================================");
    console.log(`[RESPONSE SOURCE] ${label}`);
    if (safe.intent) console.log(`[INTENT] ${safe.intent}`);
    if (safe.text) console.log(`[QUESTION] ${safe.text}`);
    if (safe.latencyMs != null) console.log(`[RESPONSE TIME] ${safe.latencyMs} ms`);
    console.log("============================================================\n");
  }

  const APPROVED_LEARN_INTENTS = new Set([
    "what_is","strengths","dose_frequency","side_effects","side_effect_help","side_effect_medicine","other_medicines","missed_dose","how_to_take","timing",
    "food","pregnancy","breastfeeding","kidney_liver","potassium","under18","stopping","co_extor","overdose","emergency","purchase","price","competitor","precautions","bp_related","alcohol","monitoring","effectiveness","onset","storage","refill","travel"
  ]);
  if (safe.matched && APPROVED_LEARN_INTENTS.has(safe.intent) && safe.text.length >= 3) {
    await addLearnedPhrase(safe.intent, safe.text);
  }
  res.json({ ok: true, storage: learningDbReady ? "postgres" : "json" });
});

function normalizeLearnText(value) {
  return String(value || "").toLowerCase()
    .replace(/[؟?۔.,!،;؛:]/g, " ")
    .replace(/\s+/g, " ").trim();
}

// Learned answers are a performance cache, not a self-editing medical source.
// Only answers produced by Claude under the locked Extor prompt can be saved,
// and only for a small approved set of general Extor intents.
app.post("/api/learn-answer", async (req, res) => {
  const body = req.body || {};
  const allowed = new Set(["what_is", "strengths", "dose_frequency", "side_effect_help", "side_effects", "side_effect_medicine", "other_medicines", "timing", "food", "missed_dose", "how_to_take", "pregnancy", "breastfeeding", "kidney_liver", "potassium", "under18", "stopping", "co_extor", "overdose", "emergency", "price", "competitor", "purchase", "precautions", "bp_related", "alcohol", "monitoring", "effectiveness", "onset", "storage", "refill", "travel", "other_extor"]);
  const question = normalizeLearnText(body.question).slice(0, 500);
  const intent = inferLearningIntent(question, body.intent);
  const answer = String(body.answer || "").replace(/<state>[\s\S]*?<\/state>/g, "").trim().slice(0, 700);
  if (looksLikeChitChatForLearning(question)) return res.json({ok:true, skipped:"chitchat"});
  if (!allowed.has(intent) || question.length < 5 || answer.length < 8) return res.status(400).json({ok:false});
  if (/\b\d{2,3}\s*(?:\/|over|by)\s*\d{2,3}\b/i.test(question) || /\b\d{2,3}\s*kg\b/i.test(question)) {
    return res.json({ok:true, skipped:"patient_specific"});
  }
  try {
    if (await dbIsReady()) {
      await learningPool.query(
        `INSERT INTO sania_learned_answers(intent,question,answer,hits,updated_at)
         VALUES($1,$2,$3,0,NOW())
         ON CONFLICT(intent,question) DO UPDATE SET answer=EXCLUDED.answer, updated_at=NOW()`,
        [intent, question, answer]
      );
    } else {
      let rows = JSON.parse(fs.readFileSync(LEARNED_ANSWERS_FILE, "utf8") || "[]");
      if (!Array.isArray(rows)) rows = [];
      const existing = rows.find(r => r.intent === intent && r.question === question);
      const row = { intent, question, answer, updatedAt:new Date().toISOString(), hits: existing ? Number(existing.hits||0) : 0 };
      rows = rows.filter(r => !(r.intent === intent && r.question === question));
      rows.push(row); rows = rows.slice(-400);
      fs.writeFileSync(LEARNED_ANSWERS_FILE, JSON.stringify(rows, null, 2));
    }
    await addLearnedPhrase(intent, question);
    console.log(`[learning-promote] intent=${intent} question="${question.slice(0,120)}" storage=${learningDbReady?'postgres':'json'}`);
    return res.json({ok:true, intent, storage:learningDbReady?"postgres":"json"});
  } catch (e) { return res.status(500).json({ok:false,error:e.message}); }
});


// --- Server-side learned answer matcher (PostgreSQL-first) ---
function lmNorm(v){
  return String(v||"").toLowerCase().replace(/[؟?۔.,!؛:()\[\]{}]/g," ").replace(/\s+/g," ").trim();
}
function lmCanon(w){
  const x=String(w||"").toLowerCase();
  const groups={
    strength:["strength","strengths","power","powers","variation","variations","variant","variants","طاقت","ورائٹی"],
    dose:["dose","dosage","خوراک"], twice:["twice","2","two","do","دو","بار","دفعہ"],
    food:["food","khana","khane","کھانا","کھانے"], water:["water","pani","پانی"], milk:["milk","doodh","دودھ"],
    miss:["miss","missed","bhool","بھول"], kidney:["kidney","kidneys","renal","gurda","gurday","گردہ","گردے"],
    liver:["liver","hepatic","jigar","جگر"], pregnant:["pregnant","pregnancy","hamal","حمل","حاملہ"],
    dizzy:["dizzy","dizziness","chakkar","چکر"], swelling:["swelling","soojan","sooj","سوجن"],
    headache:["headache","sar","sir","سر","درد"], nausea:["nausea","متلی","matli"],
    panadol:["panadol","paracetamol","acetaminophen"], ibuprofen:["ibuprofen","brufen","nsaid","nsaids"],
    aspirin:["aspirin"], potassium:["potassium","پوٹاشیم"], stop:["stop","stopping","band","بند","چھوڑ"],
    price:["price","cost","rate","قیمت","دام","pkr","روپے"], competitor:["competitor","alternative","substitute","compare","comparison","versus","exforge","avsar","amlortan","amstan","dioplus","newday","valam"],
    bp:["bp","blood","pressure","بلڈ","پریشر"]
  };
  for(const [k,a] of Object.entries(groups)) if(a.includes(x)) return k;
  return x;
}
function lmTokens(v){
  const stop=new Set(["extor","ایکسٹور","medicine","medicin","tablet","dawa","goli","دوا","گولی","kya","kia","hai","hain","he","main","mein","mujhe","mera","meri","mere","what","is","are","the","a","an","i","me","my","can","could","would","do","does","with","for","of","to","it","کیا","ہے","ہیں","میں","مجھے","میرا","میری","میرے","کے","کی","کا","کو","سے","اور"]);
  return new Set(lmNorm(v).split(/\s+/).map(lmCanon).filter(x=>x&&x.length>1&&!stop.has(x)));
}
function lmOverlap(a,b){
  const A=lmTokens(a),B=lmTokens(b); if(!A.size||!B.size) return 0;
  let n=0; for(const x of A) if(B.has(x)) n++;
  return n/Math.max(1,Math.min(A.size,B.size));
}
function lmJaccard(a,b){
  const A=lmTokens(a),B=lmTokens(b); if(!A.size||!B.size) return 0;
  let n=0; for(const x of A) if(B.has(x)) n++;
  return n/Math.max(A.size,B.size);
}
function lmConcepts(v){
  const t=lmNorm(v); const out=new Set();
  const rules={
    kidney:["kidney","kidneys","renal","gurda","gurday","گردہ","گردے"], liver:["liver","hepatic","jigar","جگر"],
    pregnancy:["pregnant","pregnancy","hamal","حمل","حاملہ"], breastfeeding:["breastfeed","breastfeeding","nursing","دودھ پلا"],
    dizziness:["dizzy","dizziness","chakkar","چکر"], swelling:["swelling","soojan","سوجن"], headache:["headache","sar dard","sir dard","سر درد"], nausea:["nausea","matli","متلی"],
    panadol:["panadol","paracetamol","acetaminophen"], ibuprofen:["ibuprofen","brufen"], aspirin:["aspirin"], potassium:["potassium","پوٹاشیم"],
    twice:["twice","two times","2 times","2 tablets","two tablets","do baar","2 baar","دو بار","دو دفعہ","دو گولی","دن میں دو"],
    missed:["missed dose","miss dose","dose miss","bhool","بھول"], food:["food","khana","khane","کھانا","کھانے","empty stomach","خالی پیٹ"],
    liquid:["water","pani","پانی","milk","doodh","دودھ"], strengths:["5/80","5 80","5/160","5 160","10/160","10 160","strength","power","variation","طاقت"],
    price:["price","cost","rate","قیمت","دام","pkr","روپے"], competitor:["competitor","alternative","substitute","compare","comparison","versus","exforge","avsar","amlortan","amstan","dioplus","newday","valam"],
    stop:["stop","stopping","band","بند","چھوڑ"], bp:["bp","blood pressure","بلڈ پریشر"]
  };
  for(const [k,a] of Object.entries(rules)) if(a.some(x=>t.includes(x))) out.add(k);
  return out;
}
function lmConceptOverlap(a,b){
  const A=lmConcepts(a),B=lmConcepts(b); if(!A.size||!B.size) return 0;
  let n=0; for(const x of A) if(B.has(x)) n++;
  return n/Math.max(1,Math.min(A.size,B.size));
}
const LM_CANONICAL_INTENTS=new Set(["what_is","strengths","dose_frequency","side_effects","missed_dose","how_to_take","timing","food","stopping","co_extor","price","competitor","pregnancy","breastfeeding","potassium","under18"]);
const LM_CONCEPT_INTENTS=new Set(["side_effect_help","side_effect_medicine","other_medicines","kidney_liver","precautions","bp_related"]);
function lmRank(question,row,intent,totalSameIntent){
  const exact=lmNorm(question)===lmNorm(row.question);
  const jac=lmJaccard(question,row.question), ov=lmOverlap(question,row.question), co=lmConceptOverlap(question,row.question);
  let ok=false, reason="";
  if(exact){ok=true;reason="exact";}
  else if(LM_CANONICAL_INTENTS.has(intent)){
    // Once the current utterance has already been confidently classified into a
    // canonical intent, one learned answer for that intent is reusable across language/rephrasing.
    if(totalSameIntent===1){ok=true;reason="single-intent-answer";}
    else if(co>0 || ov>=0.28 || jac>=0.22){ok=true;reason="canonical-intent";}
  } else if(LM_CONCEPT_INTENTS.has(intent)){
    if(co>=0.5 && (ov>=0.15 || jac>=0.12 || co>=1)){ok=true;reason="concept-match";}
    else if(ov>=0.52 || jac>=0.45){ok=true;reason="strong-wording";}
  } else if(intent==="other_extor"){
    if((co>=1 && ov>=0.35) || ov>=0.72 || jac>=0.62){ok=true;reason="strict-other";}
  } else if(ov>=0.48 || jac>=0.40){ok=true;reason="wording";}
  const score=(jac*0.35)+(ov*0.35)+(co*0.30)+(exact?1:0);
  return {ok,score,jac,ov,co,reason};
}
app.post("/api/learned-match", async (req,res)=>{
  try{
    const question=String(req.body?.question||"").trim();
    let intent=String(req.body?.intent||"").trim();
    if(intent==="unknown_extor") intent="other_extor";
    if(!question||!intent) return res.status(400).json({ok:false,error:"question and intent required"});
    // If the browser only knows this is an Extor question, let the server use
    // the complete utterance to promote it into a supervised/specific intent.
    if(intent==="other_extor"){
      const inferred = inferLearningIntent(question, null);
      if(inferred && inferred!=="other_extor") intent=inferred;
    }
    let rows=[];
    if(learningPool){
      const q=await learningPool.query(`SELECT intent,question,answer,hits,updated_at AS "updatedAt" FROM sania_learned_answers WHERE intent=$1 ORDER BY hits DESC, updated_at DESC LIMIT 100`,[intent]);
      rows=q.rows;
    }else{
      rows=readJsonSafe(LEARNED_ANSWERS_FILE,[]).filter(x=>x&&x.intent===intent);
    }
    if(!rows.length) return res.json({ok:true,hit:false,intent,count:0});
    let best=null,bestRank=null;
    for(const row of rows){ const r=lmRank(question,row,intent,rows.length); if(r.ok && (!bestRank || r.score>bestRank.score)){best=row;bestRank=r;} }
    if(!best) return res.json({ok:true,hit:false,intent,count:rows.length});
    if(learningPool){
      await learningPool.query(`UPDATE sania_learned_answers SET hits=hits+1,updated_at=NOW() WHERE intent=$1 AND question=$2`,[best.intent,best.question]);
      await learningPool.query(`INSERT INTO sania_learning_events(ts,call_id,text,intent,matched,used_claude,source) VALUES(NOW(),$1,$2,$3,true,false,'LEARNED_DB')`,[String(req.body?.callId||""),question,best.intent]);
    }
    console.log(`[learned-server-match] HIT intent=${intent} reason=${bestRank.reason} score=${bestRank.score.toFixed(2)} overlap=${bestRank.ov.toFixed(2)} concept=${bestRank.co.toFixed(2)} current="${question.slice(0,100)}" learned="${best.question.slice(0,100)}"`);
    return res.json({ok:true,hit:true,intent:best.intent,question:best.question,answer:best.answer,score:bestRank.score,overlap:bestRank.ov,conceptOverlap:bestRank.co,reason:bestRank.reason});
  }catch(e){console.error('[learned-server-match] error',e);return res.status(500).json({ok:false,error:String(e.message||e)});}
});

app.post("/api/learned-hit", async (req,res) => {
  const body=req.body||{};
  const intent=String(body.intent||"").slice(0,80);
  const question=normalizeLearnText(body.question).slice(0,500);
  const currentText=normalizeLearnText(body.currentText).slice(0,500);
  if(!intent || !question) return res.status(400).json({ok:false});
  try{
    if(await dbIsReady()){
      await learningPool.query(`UPDATE sania_learned_answers SET hits=hits+1, updated_at=NOW() WHERE intent=$1 AND question=$2`,[intent,question]);
    } else {
      let rows=[]; try{ rows=JSON.parse(fs.readFileSync(LEARNED_ANSWERS_FILE,"utf8")||"[]"); }catch(e){}
      if(Array.isArray(rows)){
        const row=rows.find(r=>r.intent===intent && normalizeLearnText(r.question)===question);
        if(row){ row.hits=Number(row.hits||0)+1; row.updatedAt=new Date().toISOString(); fs.writeFileSync(LEARNED_ANSWERS_FILE,JSON.stringify(rows,null,2)); }
      }
    }
    // The normal /api/learn event emitted after speech records source=LEARNED_DB;
    // this endpoint exists specifically to maintain the durable answer hit counter.
    console.log(`[learned-hit] intent=${intent} question="${question.slice(0,100)}" current="${currentText.slice(0,100)}"`);
    return res.json({ok:true,storage:learningDbReady?"postgres":"json"});
  }catch(e){ return res.status(500).json({ok:false,error:e.message}); }
});


function trainingAuthorized(req){
  return !!TRAINING_IMPORT_KEY && String(req.get("x-training-key")||"") === TRAINING_IMPORT_KEY;
}
function loadSupervisedTrainingPack(){
  const raw=fs.readFileSync(SUPERVISED_TRAINING_FILE,"utf8");
  return JSON.parse(raw);
}
app.get("/api/training-pack-info", async (req,res)=>{
  try{
    const pack=loadSupervisedTrainingPack();
    let imported=null;
    if(await dbIsReady()){
      const q=await learningPool.query(`SELECT version,checksum,records_total AS "recordsTotal",answers_imported AS "answersImported",phrases_imported AS "phrasesImported",imported_at AS "importedAt" FROM sania_training_packs WHERE version=$1`,[pack.version]);
      imported=q.rows[0]||null;
    }
    const modes=(pack.records||[]).reduce((a,r)=>{a[r.mode||"answer"]=(a[r.mode||"answer"]||0)+1;return a;},{});
    const intents=(pack.records||[]).reduce((a,r)=>{a[r.intent]=(a[r.intent]||0)+1;return a;},{});
    res.json({ok:true,version:pack.version,checksum:pack.checksum,records:(pack.records||[]).length,modes,intents,imported,storage:learningDbReady?"postgres":"json"});
  }catch(e){res.status(500).json({ok:false,error:e.message});}
});
app.post("/api/admin/import-supervised-training", async (req,res)=>{
  if(!trainingAuthorized(req)) return res.status(401).json({ok:false,error:"invalid training key"});
  if(!(await dbIsReady())) return res.status(503).json({ok:false,error:"PostgreSQL required for supervised import"});
  try{
    const pack=loadSupervisedTrainingPack();
    const dryRun=!!req.body?.dryRun;
    const stats={version:pack.version,recordsTotal:0,answersImported:0,phrasesImported:0,phraseOnly:0,skipped:0};
    for(const r of (pack.records||[])){
      const q=normalizeLearnText(r.question).slice(0,500);
      const intent=normalizeIntentName(r.intent);
      const answer=String(r.answer||"").replace(/<state>[\s\S]*?<\/state>/g,"").trim().slice(0,700);
      if(!q || !CANONICAL_LEARN_INTENTS.has(intent) || looksLikeChitChatForLearning(q)){stats.skipped++;continue;}
      stats.recordsTotal++;
      if(!dryRun){
        await learningPool.query(`INSERT INTO sania_learned_phrases(intent,phrase) VALUES($1,$2) ON CONFLICT(intent,phrase) DO NOTHING`,[intent,q]);
      }
      stats.phrasesImported++;
      if(String(r.mode||"answer")==="phrase_only"){stats.phraseOnly++;continue;}
      if(answer.length<8){stats.skipped++;continue;}
      if(!dryRun){
        await learningPool.query(
          `INSERT INTO sania_learned_answers(intent,question,answer,hits,updated_at)
           VALUES($1,$2,$3,0,NOW())
           ON CONFLICT(intent,question) DO UPDATE SET
             answer=EXCLUDED.answer,
             hits=sania_learned_answers.hits,
             updated_at=NOW()`,
          [intent,q,answer]
        );
      }
      stats.answersImported++;
    }
    if(!dryRun){
      await learningPool.query(
        `INSERT INTO sania_training_packs(version,checksum,records_total,answers_imported,phrases_imported,imported_at)
         VALUES($1,$2,$3,$4,$5,NOW())
         ON CONFLICT(version) DO UPDATE SET checksum=EXCLUDED.checksum,records_total=EXCLUDED.records_total,
         answers_imported=EXCLUDED.answers_imported,phrases_imported=EXCLUDED.phrases_imported,imported_at=NOW()`,
        [pack.version,pack.checksum||"",stats.recordsTotal,stats.answersImported,stats.phrasesImported]
      );
      console.log(`[supervised-training] IMPORTED version=${pack.version} answers=${stats.answersImported} phrases=${stats.phrasesImported} phraseOnly=${stats.phraseOnly}`);
    }else{
      console.log(`[supervised-training] DRY RUN version=${pack.version} answers=${stats.answersImported} phrases=${stats.phrasesImported}`);
    }
    res.json({ok:true,dryRun,storage:"postgres",...stats});
  }catch(e){console.error("[supervised-training] failed",e);res.status(500).json({ok:false,error:e.message});}
});

app.post("/api/learning/reclassify", async (req,res) => {
  const result = await reclassifyExistingOtherExtor();
  res.status(result.ok===false ? 500 : 200).json(result);
});

app.get("/api/learned-answers", async (req, res) => {
  try {
    if (await dbIsReady()) {
      const q = await learningPool.query(`SELECT intent,question,answer,hits,updated_at AS "updatedAt" FROM sania_learned_answers ORDER BY updated_at DESC LIMIT 1000`);
      return res.json({ok:true, storage:"postgres", answers:q.rows});
    }
    const rows = JSON.parse(fs.readFileSync(LEARNED_ANSWERS_FILE, "utf8") || "[]");
    res.json({ok:true, storage:"json", answers:Array.isArray(rows) ? rows : []});
  } catch (e) { res.json({ok:true, storage:learningDbReady?"postgres":"json", answers:[]}); }
});

app.get("/api/learned-phrases", async (req, res) => {
  try {
    if (await dbIsReady()) {
      const q = await learningPool.query(`SELECT intent, phrase FROM sania_learned_phrases ORDER BY id ASC`);
      const learned={};
      for(const row of q.rows){ learned[row.intent]=learned[row.intent]||[]; learned[row.intent].push(row.phrase); }
      return res.json({ok:true, storage:"postgres", learned});
    }
    let learned = {}; try { learned = JSON.parse(fs.readFileSync(LEARNED_PHRASES_FILE, "utf8") || "{}"); } catch (e) {}
    res.json({ ok: true, storage:"json", learned });
  } catch(e){ res.json({ok:true, learned:{}}); }
});

app.get("/api/learning-summary", async (req, res) => {
  try {
    let learned={}, recent=[], answers=[];
    if (await dbIsReady()) {
      const [phrasesQ, eventsQ, answersQ] = await Promise.all([
        learningPool.query(`SELECT intent, phrase FROM sania_learned_phrases ORDER BY id ASC`),
        learningPool.query(`SELECT ts,call_id AS "callId",text,intent,matched,used_claude AS "usedClaude",latency_ms AS "latencyMs",source FROM sania_learning_events ORDER BY ts DESC LIMIT 50`),
        learningPool.query(`SELECT intent,question,answer,hits,updated_at AS "updatedAt" FROM sania_learned_answers ORDER BY updated_at DESC LIMIT 1000`)
      ]);
      for(const row of phrasesQ.rows){ learned[row.intent]=learned[row.intent]||[]; learned[row.intent].push(row.phrase); }
      recent=eventsQ.rows; answers=answersQ.rows;
    } else {
      try { learned = JSON.parse(fs.readFileSync(LEARNED_PHRASES_FILE, "utf8") || "{}"); } catch (e) {}
      try { const lines = fs.readFileSync(LEARNING_LOG, "utf8").trim().split("\n").filter(Boolean).slice(-50); recent = lines.map(x => { try { return JSON.parse(x); } catch (e) { return null; } }).filter(Boolean).reverse(); } catch (e) {}
      try { answers = JSON.parse(fs.readFileSync(LEARNED_ANSWERS_FILE, "utf8") || "[]"); } catch (e) {}
      if (!Array.isArray(answers)) answers = [];
    }
    const learnedAnswerCounts = {};
    for (const row of answers) { const k=String(row && row.intent || "unknown"); learnedAnswerCounts[k]=(learnedAnswerCounts[k]||0)+1; }
    const quality = { learnedAnswers:answers.length, learnedPhrases:Object.values(learned).reduce((n,v)=>n+(Array.isArray(v)?v.length:0),0), recentClaudeFallbacks:recent.filter(x=>x&&x.usedClaude).length, recentLearnedDbHits:recent.filter(x=>x&&String(x.source||"").includes("LEARNED_DB")).length, recentUnknown:recent.filter(x=>x&&(x.intent==="unknown_extor"||x.intent==="other_extor")).length };
    res.json({ ok:true, storage:learningDbReady?"postgres":"json", learnedPhraseCounts:Object.fromEntries(Object.entries(learned).map(([k,v])=>[k,Array.isArray(v)?v.length:0])), learnedAnswerCounts, quality, recent });
  } catch(e){ res.status(500).json({ok:false,error:e.message}); }
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
  upliftVoice: UPLIFT_TTS_VOICE,
  learningStorage: learningDbReady ? "postgres" : "json",
  hasDatabaseUrl: !!DATABASE_URL
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
