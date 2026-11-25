// server.js – Hold On backend
// Combines DetectifAI Aletheia (deepfake voice detection) + pressure-tactic
// keyword heuristic + Telegram guardian alerts.

import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import { createHash, randomUUID } from 'crypto';
import { readFile, unlink, mkdir, stat } from 'fs/promises';
import { existsSync } from 'fs';
import { join, extname } from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { fileURLToPath } from 'url';
import FormData from 'form-data';
import fetch from 'node-fetch';

const execFileAsync = promisify(execFile);
const __dirname = fileURLToPath(new URL('.', import.meta.url));

// ─── Config ───────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
const DETECTIF_API_KEY = process.env.DETECTIF_API_KEY;
const STT_URL = process.env.STT_URL || 'https://api.groq.com/openai/v1/audio/transcriptions';
const STT_MODEL = process.env.STT_MODEL || 'whisper-large-v3';
const STT_KEY = process.env.STT_KEY;
const TG_TOKEN = process.env.TG_TOKEN;
const TG_CHAT_ID = process.env.TG_CHAT_ID;
const WA_TOKEN = process.env.WA_TOKEN;
const WA_PHONE_NUMBER_ID = process.env.WA_PHONE_NUMBER_ID;
const WA_TO = process.env.WA_TO;
const WA_VERIFY_TOKEN = process.env.WA_VERIFY_TOKEN || 'holdon_verify_token';
const GREEN_API_URL = process.env.GREEN_API_URL;
const GREEN_API_ID_INSTANCE = process.env.GREEN_API_ID_INSTANCE;
const GREEN_API_TOKEN = process.env.GREEN_API_TOKEN;

const TMP_DIR = join(__dirname, 'tmp');
const ALLOWED_EXT = new Set(['.webm', '.mp3', '.m4a', '.wav', '.ogg', '.opus', '.aac', '.flac']);
const MIN_DURATION = 4;
const MAX_DURATION = 120;

// ─── Find ffmpeg ──────────────────────────────────────────────────
let FFMPEG_PATH = 'ffmpeg';
try {
  // Try system ffmpeg first
  await execFileAsync('ffmpeg', ['-version']);
} catch {
  // Fall back to ffmpeg-static npm package
  try {
    const ffmpegStatic = await import('ffmpeg-static');
    FFMPEG_PATH = ffmpegStatic.default;
    console.log(`[ffmpeg] Using ffmpeg-static: ${FFMPEG_PATH}`);
  } catch {
    console.error('╔═══════════════════════════════════════════════════╗');
    console.error('║  ffmpeg not found!                                ║');
    console.error('║  Install ffmpeg: https://ffmpeg.org/download.html ║');
    console.error('║  Or: npm install ffmpeg-static                    ║');
    console.error('╚═══════════════════════════════════════════════════╝');
    process.exit(1);
  }
}

// ─── Express setup ────────────────────────────────────────────────
const app = express();
await mkdir(TMP_DIR, { recursive: true });

const upload = multer({
  dest: TMP_DIR,
  limits: { fileSize: 25 * 1024 * 1024 }, // 25 MB
  fileFilter(_, file, cb) {
    const ext = extname(file.originalname).toLowerCase();
    // Also accept webm from MediaRecorder (may come as audio/webm with no ext match)
    if (ALLOWED_EXT.has(ext) || file.mimetype.startsWith('audio/')) {
      cb(null, true);
    } else {
      cb(new Error(`Unsupported audio format: ${ext || file.mimetype}`));
    }
  }
});

app.use(express.json());
app.use(express.static(join(__dirname, 'public')));

// ─── In-memory Aletheia cache (SHA-256 of WAV → result) ──────────
const aletheiaCache = new Map();

// ─── ffmpeg: convert to 16kHz mono WAV ───────────────────────────
async function convertToWav(inputPath) {
  const outPath = inputPath + '.wav';
  try {
    await execFileAsync(FFMPEG_PATH, [
      '-y', '-i', inputPath,
      '-af', 'highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11',
      '-ar', '16000', '-ac', '1', '-f', 'wav',
      outPath
    ], { timeout: 30_000 });
  } catch (err) {
    throw new Error(`ffmpeg conversion failed: ${err.message}`);
  }
  return outPath;
}

// ─── ffprobe: get duration ───────────────────────────────────────
async function getDuration(filePath) {
  try {
    // Use ffmpeg's ffprobe or parse ffmpeg output
    const ffprobePath = FFMPEG_PATH.replace(/ffmpeg(\.exe)?$/i, 'ffprobe$1');
    const { stdout } = await execFileAsync(ffprobePath, [
      '-v', 'error', '-show_entries', 'format=duration',
      '-of', 'csv=p=0', filePath
    ], { timeout: 10_000 });
    return parseFloat(stdout.trim());
  } catch {
    // Fallback: estimate from file size (16kHz mono 16-bit = 32000 bytes/sec)
    try {
      const stats = await stat(filePath);
      // WAV header ~44 bytes
      return Math.max(0, (stats.size - 44) / 32000);
    } catch {
      return 0;
    }
  }
}

// ─── Aletheia: deepfake voice detection ──────────────────────────
// Mapper for Aletheia's actual response format (verified against live API).
// Response shape:
// {
//   request_id: "req_...",
//   operation_id: "op_...",
//   object: "deepfake.detection",
//   result: {
//     verdict: "real" | "fake",
//     score: 0.0–1.0 (higher = more likely synthetic),
//     duration_seconds: number,
//     scored_seconds: number,
//     segments: [{ start, end, score, verdict }]
//   }
// }
// Error shape: { error: { type, code, message, param }, request_id }
function readVoice(raw) {
  // Log the raw response for debugging
  console.log('[Aletheia] Raw response:', JSON.stringify(raw, null, 2));

  // Check for API-level errors
  if (raw?.error) {
    throw new Error(`Aletheia: ${raw.error.message || raw.error.code || 'Unknown error'}`);
  }

  const data = raw?.result;
  if (!data) {
    // Fallback: try top-level fields in case format changes
    const fallbackVerdict = raw?.verdict || '';
    const fallbackScore = raw?.score ?? null;
    return {
      synthetic: /fake/i.test(fallbackVerdict),
      verdict: /fake/i.test(fallbackVerdict) ? 'synthetic' : 'human',
      score: typeof fallbackScore === 'number' ? fallbackScore : null
    };
  }

  const verdictRaw = (data.verdict || data.label || '').toLowerCase();
  const score = typeof data.score === 'number' ? data.score : null;
  const synthetic = typeof data.is_synthetic === 'boolean'
    ? data.is_synthetic
    : (verdictRaw === 'fake' || verdictRaw === 'synthetic');
  const verdict = synthetic ? 'synthetic' : 'human';

  return { synthetic, verdict, score };
}

async function callAletheia(wavPath) {
  if (!DETECTIF_API_KEY) {
    throw new Error('DETECTIF_API_KEY is not set in .env');
  }

  // Check cache
  const wavBuffer = await readFile(wavPath);
  const hash = createHash('sha256').update(wavBuffer).digest('hex');

  if (aletheiaCache.has(hash)) {
    console.log('[Aletheia] Cache hit for', hash.slice(0, 12));
    return aletheiaCache.get(hash);
  }

  const form = new FormData();
  form.append('file', wavBuffer, { filename: 'clip.wav', contentType: 'audio/wav' });

  let res;
  try {
    res = await fetch('https://platform.detectif.ai/v1/deepfake/detect', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${DETECTIF_API_KEY}`,
        'Idempotency-Key': randomUUID(),
        ...form.getHeaders()
      },
      body: form
    });
  } catch (err) {
    throw new Error(`Aletheia API unreachable: ${err.message}`);
  }

  // Parse body for all responses (Aletheia returns JSON even on errors)
  const json = await res.json().catch(() => null);

  if (!res.ok) {
    // Handle structured error from Aletheia
    if (json?.error) {
      const errMsg = json.error.message || json.error.code;
      // 422 with insufficient_speech means audio has <2s of clear speech;
      // degrade gracefully so STT + pressure check still run.
      if (res.status === 422 && json.error.code === 'insufficient_speech') {
        console.warn('[Aletheia] Insufficient speech:', errMsg);
        return {
          synthetic: false,
          verdict: 'unverified',
          score: null,
          warning: `Voice check skipped (${errMsg})`
        };
      }
      throw new Error(`Aletheia: ${errMsg}`);
    }
    const messages = {
      401: 'Invalid API key. Check your DETECTIF_API_KEY in .env.',
      402: 'Aletheia credits exhausted. Contact DetectifAI to top up.',
      429: 'Too many requests – please wait a moment and try again.',
    };
    const msg = messages[res.status] || `Aletheia API error ${res.status}`;
    throw new Error(msg);
  }

  const result = readVoice(json);

  aletheiaCache.set(hash, result);
  return result;
}

// ─── Speech-to-text via Whisper-compatible API ───────────────────
async function transcribe(wavPath, lang = 'en') {
  if (!STT_KEY) {
    return { transcript: null, error: 'STT_KEY not configured' };
  }

  const wavBuffer = await readFile(wavPath);
  const form = new FormData();
  form.append('file', wavBuffer, { filename: 'clip.wav', contentType: 'audio/wav' });
  form.append('model', STT_MODEL);
  if (lang) form.append('language', lang);

  let res;
  try {
    res = await fetch(STT_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${STT_KEY}`,
        ...form.getHeaders()
      },
      body: form
    });
  } catch (err) {
    return { transcript: null, error: `STT unreachable: ${err.message}` };
  }

  if (!res.ok) {
    const body = await res.text().catch(() => '');
    return { transcript: null, error: `STT error ${res.status}: ${body.slice(0, 200)}` };
  }

  const json = await res.json();
  return { transcript: json.text || json.transcript || '', error: null };
}

// ─── Pressure-tactic keyword heuristic ───────────────────────────
// NOTE: This is a simple keyword/phrase heuristic, NOT NLP or ML.
// It will have false positives and negatives. See README for limitations.
const PRESSURE_KEYWORDS = {
  urgency: {
    en: ['immediately', 'right now', 'hurry', 'urgent', 'quickly', 'fast', 'rush', 'today only', 'last chance', 'now or never', "don't wait", 'act now', 'time is running out', 'expires', 'deadline', 'within 1 hour', 'within one hour', 'need help', 'send help', 'help me', 'save me', 'in trouble', 'emergency', 'abhi', 'jaldi', 'turant'],
    hi: ['तुरंत', 'अभी', 'जल्दी', 'फौरन', 'अभी के अभी', 'देर मत करो', 'समय निकल रहा है', 'आज ही', 'अभी करो', 'जल्दी करो', 'रुको मत', 'मदद करो', 'बचाओ', 'मुसीबत में'],
    kn: ['ಈಗಲೇ', 'ತಕ್ಷಣ', 'ಬೇಗ', 'ಬೇಗನೆ', 'ಆತುರ', 'ತಡ ಮಾಡಬೇಡಿ', 'ಹೊತ್ತು ಮೀರುತ್ತಿದೆ', 'ಸಹಾಯ ಮಾಡಿ', 'ಕಾಪಾಡಿ'],
    es: ['inmediatamente', 'ahora mismo', 'apresúrate', 'urgente', 'rápido', 'pronto', 'hoy sólo', 'última oportunidad', 'ahora o nunca', 'no esperes', 'actúa ahora', 'el tiempo se acaba', 'fecha límite', 'en una hora', 'necesito ayuda', 'ayuda', 'salva me', 'en problemas', 'emergencia'],
    fr: ['immédiatement', 'tout de suite', 'dépêche-toi', 'urgent', 'rapidement', 'vite', 'aujourd\'hui seulement', 'dernière chance', 'maintenant ou jamais', "n'attends pas", 'agis maintenant', 'le temps presse', 'date limite', 'dans une heure', 'besoin d\'aide', 'aide', 'sauve-moi', 'en difficulté', 'urgence'],
    zh: ['立刻', '马上', '赶紧', '紧急', '快速', '尽快', '仅今日', '最后机会', '现在或永不', '别等了', '立即行动', '时间不多了', '截止日期', '一小时内', '需要帮助', '帮助', '救命', '陷入困境', '紧急情况']
  },
  secrecy: {
    en: ['don\'t tell', 'do not tell', 'keep it secret', 'between us', 'confidential', 'don\'t share', 'don\'t inform', 'keep quiet', 'secret', 'nobody should know', "don\'t mention", 'just between you and me', 'kisi ko mat batao', 'kisi ko mat batana'],
    hi: ['किसी को मत बताना', 'किसी को मत बताओ', 'गुप्त', 'छुपाकर', 'बीच में कोई नहीं', 'किसी से मत कहना', 'चुप रहो', 'रहस्य', 'किसी को पता नहीं चलना चाहिए'],
    kn: ['ಯಾರಿಗೂ ಹೇಳಬೇಡಿ', 'ರಹಸ್ಯ', 'ಗುಟ್ಟು', 'ಯಾರಿಗೂ ತಿಳಿಸಬೇಡಿ', 'ಬಾಯಿ ಮುಚ್ಚಿಕೊಂಡಿರಿ'],
    es: ['no lo cuentes', 'no lo digas', 'mantenerlo secreto', 'entre nosotros', 'confidencial', 'no compartir', 'no informar', 'mantener silencio'],
    fr: ['ne le dites pas', 'ne le révélez pas', 'garder secret', 'entre nous', 'confidentiel', 'ne partagez pas', 'ne pas informer', 'garder le silence'],
    zh: ['不要透露', '不要告诉', '保密', '我们之间', '机密', '不分享', '不通报', '保持沉默']
  },
  money: {
    en: ['send money', 'transfer', 'UPI', 'gift card', 'account number', 'OTP', 'payment', 'rupees', 'bank account', 'wire', 'cash', 'google pay', 'gpay', 'phone pe', 'phonepe', 'paytm', 'neft', 'imps', 'rtgs', 'bitcoin', 'crypto', 'one-time password', 'send 20000', 'send 50000', 'credit card', 'debit card', 'CVV', 'pin number', 'bank details', 'ransom'],
    hi: ['पैसे भेजो', 'ट्रांसफर', 'UPI', 'गिफ्ट कार्ड', 'अकाउंट नंबर', 'ओटीपी', 'OTP', 'भुगतान', 'रुपये', 'बैंक खाता', 'पेटीएम', 'गूगल पे', 'फोन पे', 'पैसा', 'रकम', 'फिरौती'],
    kn: ['ಹಣ ಕಳುಹಿಸಿ', 'ವರ್ಗಾವಣೆ', 'ಯುಪಿಐ', 'ಗಿಫ್ಟ್ ಕಾರ್ಡ್', 'ಖಾತೆ ಸಂಖ್ಯೆ', 'ಒಟಿಪಿ', 'ಪಾವತಿ', 'ರೂಪಾಯಿ', 'ಬ್ಯಾಂಕ್ ಖಾತೆ'],
    es: ['enviar dinero', 'transferir', 'UPI', 'tarjeta de regalo', 'número de cuenta', 'OTP', 'pago', 'rupees', 'cuenta bancaria', 'efectivo', 'Google Pay', 'Paytm', 'transferencia'],
    fr: ['envoyer de l\'argent', 'transférer', 'UPI', 'carte cadeau', 'numéro de compte', 'OTP', 'paiement', 'rupees', 'compte bancaire', 'espèces', 'Google Pay', 'Paytm'],
    zh: ['汇款', '转账', 'UPI', '礼品卡', '账号', '一次性密码', '支付', '卢比', '银行账户', '现金', 'Google Pay', 'Paytm']
  },
  threat: {
    en: ['arrest', 'arrested', 'police', 'CBI', 'customs', 'warrant', 'digital arrest', 'case registered', 'FIR', 'court', 'jail', 'prison', 'legal action', 'law enforcement', 'investigation', 'ED', 'enforcement directorate', 'income tax', 'raid', 'narcotics', 'NCB', 'suspend your account', 'block your account', 'seize', 'penalty', 'fine', 'prosecution'],
    hi: ['गिरफ्तार', 'गिरफ्तारी', 'पुलिस', 'सीबीआई', 'CBI', 'कस्टम', 'वारंट', 'डिजिटल अरेस्ट', 'केस दर्ज', 'FIR', 'कोर्ट', 'जेल', 'कानूनी कार्रवाई', 'जांच', 'ईडी', 'आयकर', 'छापा', 'NCB', 'खाता बंद', 'जब्त', 'जुर्माना'],
    kn: ['ಬಂಧನ', 'ಪೊಲೀಸ್', 'ಸಿಬಿಐ', 'ಕಸ್ಟಮ್ಸ್', 'ವಾರಂಟ್', 'ಡಿಜಿಟಲ್ ಅರೆಸ್ಟ್', 'ಎಫ್ಐಆರ್', 'ನ್ಯಾಯಾಲಯ', 'ಜೈಲು', 'ಕಾನೂನು ಕ್ರಮ'],
    es: ['arresto', 'policía', 'orden de captura', 'investigación', 'sanción', 'banco bloqueado', 'incautar', 'multa', 'enjuiciamiento'],
    fr: ['arrêt', 'police', 'mandat d\'arrêt', 'enquête', 'sanction', 'compte bloqué', 'confiscation', 'amende', 'poursuite'],
    zh: ['逮捕', '警察', '逮捕令', '调查', '处罚', '账户冻结', '没收', '罚款', '起诉']
  },
  distress: {
    en: ['kidnapped', 'kidnap', 'abducted', 'hostage', 'trapped', 'stuck', "don\'t know where i am", 'accident', 'hospital', 'injured', 'bleeding', 'danger', 'locked up', 'hurt'],
    hi: ['किडनैप', 'अपहरण', 'फंसा हुआ', 'फंसी हुई', 'एक्सीडेंट', 'अस्पताल', 'चोट', 'खतरा', 'बंद कर दिया'],
    kn: ['ಅಪಹರಣ', 'ಸಿಲುಕಿಕೊಂಡಿದ್ದೇನೆ', 'ಅಪಘಾತ', 'ಆಸ್ಪತ್ರೆ', 'ಗಾಯ', 'ಅಪಾಯ'],
    es: ['secuestrado', 'rapto', 'secuestrada', 'encerrado', 'accidente', 'hospital', 'herido', 'sangrado', 'peligro', 'encerrado'],
    fr: ['enlevé', 'kidnappe', 'enlevée', 'piégé', 'accident', 'hôpital', 'blessé', 'saignement', 'danger', 'enfermé'],
    zh: ['被绑架', '绑架', '人质', '困住', '事故', '医院', '受伤', '出血', '危险', '被锁住']
  }
}



function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function matchesPhrase(text, lowerText, phrase) {
  // Use word boundaries for pure ASCII phrases so "ED" doesn't match "need"
  if (/^[\x00-\x7F]+$/.test(phrase)) {
    return new RegExp(`\\b${escapeRegex(phrase)}\\b`, 'i').test(text);
  }
  return lowerText.includes(phrase.toLowerCase());
}

function checkPressure(text) {
  if (!text) return { categories: {}, count: 0, matches: {} };

  const lower = text.toLowerCase();
  const categories = {};
  const matches = {};

  for (const [category, langPhrases] of Object.entries(PRESSURE_KEYWORDS)) {
    const found = [];
    for (const phrases of Object.values(langPhrases)) {
      for (const phrase of phrases) {
        if (matchesPhrase(text, lower, phrase)) {
          found.push(phrase);
        }
      }
    }
    if (found.length > 0) {
      categories[category] = true;
      // De-duplicate
      matches[category] = [...new Set(found)];
    }
  }

  return {
    categories,
    count: Object.keys(categories).length,
    matches
  };
}

// ─── Risk level logic ────────────────────────────────────────────
// "high" = synthetic voice AND ≥2 pressure categories
// "warn" = synthetic voice OR ≥2 pressure categories
// "calm" = otherwise
// The app NEVER says "safe".
const CRITICAL_PHRASES = new Set([
  'kidnapped', 'kidnap', 'abducted', 'hostage', 'ransom', 'digital arrest',
  'किडनैप', 'अपहरण', 'फिरौती', 'डिजिटल अरेस्ट', 'ಅಪಹರಣ', 'ಡಿಜಿಟಲ್ ಅರೆಸ್ಟ್'
]);

function riskLevel(voice, pressure) {
  const isSynthetic = voice.synthetic;
  const highPressure = pressure.count >= 2;
  const hasCritical = Object.values(pressure.matches || {})
    .flat()
    .some(p => CRITICAL_PHRASES.has(p.toLowerCase()));

  if (isSynthetic && (highPressure || hasCritical)) return 'high';
  if (isSynthetic || highPressure || hasCritical) return 'warn';
  return 'calm';
}

// ─── Guardian Alerts (Telegram + WhatsApp) ──────────────────────
function buildAlertMessage(level, voice, pressure, transcript) {
  const categoryList = Object.keys(pressure.categories).join(', ') || 'none';
  const excerpt = transcript
    ? (transcript.length > 300 ? transcript.slice(0, 300) + '…' : transcript)
    : '(no transcript)';

  const voiceLine = voice.verdict === 'text-only'
    ? 'Text message check (no audio)'
    : `${voice.verdict}${voice.score !== null ? ` (score: ${voice.score})` : ''}`;

  return [
    `🚨 *Hold On – ${level.toUpperCase()} ALERT*`,
    '',
    `*Check:* ${voiceLine}`,
    `*Pressure tactics:* ${categoryList}`,
    '',
    `*Message / Transcript excerpt:*`,
    excerpt,
    '',
    `⚠️ Please check on your family member. This is an automated alert from Hold On.`
  ].join('\n');
}

async function sendTelegramAlert(level, alertText, wavPath) {
  if (!TG_TOKEN || !TG_CHAT_ID) {
    return { channel: 'Telegram', sent: false, reason: 'not configured' };
  }

  try {
    const textRes = await fetch(`https://api.telegram.org/bot${TG_TOKEN}/sendMessage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_id: TG_CHAT_ID,
        text: alertText,
        parse_mode: 'Markdown'
      })
    });

    if (!textRes.ok) {
      const body = await textRes.text().catch(() => '');
      console.error('[Telegram] Text message failed:', textRes.status, body);
      return { channel: 'Telegram', sent: false, reason: `HTTP ${textRes.status}` };
    }

    if (wavPath && existsSync(wavPath)) {
      const audioBuffer = await readFile(wavPath);
      const form = new FormData();
      form.append('chat_id', TG_CHAT_ID);
      form.append('document', audioBuffer, { filename: 'suspicious_clip.wav', contentType: 'audio/wav' });
      form.append('caption', `Hold On alert – ${level} risk`);

      const audioRes = await fetch(`https://api.telegram.org/bot${TG_TOKEN}/sendDocument`, {
        method: 'POST',
        headers: form.getHeaders(),
        body: form
      });

      if (!audioRes.ok) {
        const body = await audioRes.text().catch(() => '');
        console.error('[Telegram] Audio send failed:', audioRes.status, body);
      }
    }

    return { channel: 'Telegram', sent: true, reason: null };
  } catch (err) {
    console.error('[Telegram] Alert error:', err.message);
    return { channel: 'Telegram', sent: false, reason: err.message };
  }
}

async function sendWhatsAppText(to, body) {
  // GREEN-API sendMessage endpoint
  const endpoint = `${GREEN_API_URL}/waInstance${GREEN_API_ID_INSTANCE}/sendMessage/${GREEN_API_TOKEN}`;
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      chatId: `${WA_TO}@c.us`,
      message: body
    })
  });
  if (!res.ok) {
    const errText = await res.text().catch(() => '');
    throw new Error(`GREEN-API HTTP ${res.status}: ${errText.slice(0, 150)}`);
  }
}
// Legacy Meta API block removed

async function sendWhatsAppAlert(level, alertText, wavPath) {
  // Ensure GREEN-API configuration is present
  if (!GREEN_API_URL || !GREEN_API_ID_INSTANCE || !GREEN_API_TOKEN || !WA_TO) {
    return { channel: 'WhatsApp', sent: false, reason: 'not configured' };
  }

  try {
    // Send text alert via GREEN-API
    await sendWhatsAppText(WA_TO, alertText);

    // If an audio file is provided, upload via GREEN-API sendFileByUpload endpoint
    if (wavPath && existsSync(wavPath)) {
      const audioBuffer = await readFile(wavPath);
      const form = new FormData();
      form.append('chatId', `${WA_TO}@c.us`);
      form.append('file', audioBuffer, { filename: 'suspicious_clip.wav', contentType: 'audio/wav' });
      // Optional caption can be added if needed
      // form.append('caption', `Hold On alert – ${level} risk`);

      const uploadEndpoint = `${GREEN_API_URL}/waInstance${GREEN_API_ID_INSTANCE}/sendFileByUpload/${GREEN_API_TOKEN}`;
      const res = await fetch(uploadEndpoint, {
        method: 'POST',
        headers: form.getHeaders(),
        body: form
      });
      if (!res.ok) {
        const errText = await res.text().catch(() => '');
        console.error('[WhatsApp] Audio upload failed:', res.status, errText);
      }
    }
    return { channel: 'WhatsApp', sent: true, reason: null };
  } catch (err) {
    console.error('[WhatsApp] Alert error:', err.message);
    return { channel: 'WhatsApp', sent: false, reason: err.message };
  }

  if (!GREEN_API_URL || !GREEN_API_ID_INSTANCE || !GREEN_API_TOKEN || !WA_TO) {
    return { channel: 'WhatsApp', sent: false, reason: 'not configured' };
  }

  try {
    await sendWhatsAppText(WA_TO, alertText);

    if (wavPath && existsSync(wavPath)) {
      const audioBuffer = await readFile(wavPath);
      const form = new FormData();
      form.append('messaging_product', 'whatsapp');
      form.append('file', audioBuffer, { filename: 'suspicious_clip.wav', contentType: 'audio/wav' });
      const mediaRes = await fetch(`https://graph.facebook.com/v21.0/${WA_PHONE_NUMBER_ID}/media`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${WA_TOKEN}`, ...form.getHeaders() },
        body: form
      });
      if (mediaRes.ok) {
        const mediaJson = await mediaRes.json();
        if (mediaJson?.id) {
          await fetch(`https://graph.facebook.com/v21.0/${WA_PHONE_NUMBER_ID}/messages`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${WA_TOKEN}`, 'Content-Type': 'application/json' },
            body: JSON.stringify({
              messaging_product: 'whatsapp',
              to: WA_TO,
              type: 'document',
              document: { id: mediaJson.id, filename: 'suspicious_clip.wav', caption: `Hold On alert – ${level} risk` }
            })
          });
        }
      }
    }

    return { channel: 'WhatsApp', sent: true, reason: null };
  } catch (err) {
    console.error('[WhatsApp] Alert error:', err.message);
    return { channel: 'WhatsApp', sent: false, reason: err.message };
  }
}

async function sendGuardianAlert(level, voice, pressure, transcript, wavPath) {
  if (level === 'calm') {
    return { sent: false, reason: 'Level is calm – no alert needed' };
  }

  if ((!TG_TOKEN || !TG_CHAT_ID) && (!WA_TOKEN || !WA_PHONE_NUMBER_ID || !WA_TO)) {
    return { sent: false, reason: 'No guardian channel configured (set TG_TOKEN/TG_CHAT_ID or WA_TOKEN/WA_PHONE_NUMBER_ID/WA_TO)' };
  }

  const alertText = buildAlertMessage(level, voice, pressure, transcript);
  const results = await Promise.all([
    sendTelegramAlert(level, alertText, wavPath),
    sendWhatsAppAlert(level, alertText, wavPath)
  ]);

  const sentChannels = results.filter(r => r.sent).map(r => r.channel);
  const failed = results.filter(r => !r.sent && r.reason !== 'not configured');

  if (sentChannels.length > 0) {
    const note = failed.length > 0
      ? `Sent via ${sentChannels.join(' & ')} (${failed.map(f => `${f.channel}: ${f.reason}`).join('; ')})`
      : `Sent via ${sentChannels.join(' & ')}`;
    return { sent: true, reason: note };
  }

  return {
    sent: false,
    reason: failed.map(f => `${f.channel}: ${f.reason}`).join('; ') || 'Alert could not be sent'
  };
}

// ─── Cleanup helper ──────────────────────────────────────────────
async function cleanup(...paths) {
  for (const p of paths) {
    try { if (p) await unlink(p); } catch { /* already gone */ }
  }
}

// ─── Main analysis endpoint ──────────────────────────────────────
app.post('/api/analyze', upload.single('audio'), async (req, res) => {
  const uploadedPath = req.file?.path;
  let wavPath = null;

  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No audio file uploaded.' });
    }

    const lang = req.body?.lang || 'en';

    // 1. Convert to WAV
    wavPath = await convertToWav(uploadedPath);

    // 2. Check duration
    const duration = await getDuration(wavPath);
    console.log(`[Duration] ${duration.toFixed(1)}s`);
    if (duration < MIN_DURATION) {
      return res.status(400).json({
        error: `Clip too short (${duration.toFixed(1)}s). Aletheia needs at least ${MIN_DURATION} seconds.`
      });
    }
    if (duration > MAX_DURATION) {
      return res.status(400).json({
        error: `Clip too long (${duration.toFixed(1)}s). Maximum is ${MAX_DURATION} seconds.`
      });
    }

    // 3. Run Aletheia + STT in parallel
    const [voice, sttResult] = await Promise.all([
      callAletheia(wavPath),
      transcribe(wavPath, lang)
    ]);

    const transcript = sttResult.transcript;
    const sttError = sttResult.error;

    // 4. Pressure check on transcript
    const pressure = checkPressure(transcript || '');

    // 5. Determine risk level
    const level = riskLevel(voice, pressure);

    // 6. Guardian alert (non-blocking – errors don't break the response)
    const alert = await sendGuardianAlert(level, voice, pressure, transcript, wavPath).catch(err => ({
      sent: false, reason: `Alert error: ${err.message}`
    }));

    // 7. Respond
    res.json({
      level,
      checkedAt: new Date().toISOString(),
      voice: {
        verdict: voice.verdict,
        score: voice.score,
        synthetic: voice.synthetic,
        warning: voice.warning || null
      },
      pressure: {
        count: pressure.count,
        categories: pressure.categories,
        matches: pressure.matches
      },
      transcript: transcript || null,
      sttError: sttError || null,
      alert: {
        sent: alert.sent,
        reason: alert.reason || null
      },
      disclaimer: 'These results are evidence, not proof.'
    });

  } catch (err) {
    console.error('[Analyze] Error:', err);
    res.status(err.message.includes('API key') ? 401 : 500).json({
      error: err.message || 'Analysis failed'
    });
  } finally {
    // Privacy: delete temp files
    await cleanup(uploadedPath, wavPath);
  }
});

// ─── Suspicious text message endpoint ────────────────────────────
app.post('/api/check-text', async (req, res) => {
  try {
    const text = (req.body?.text || '').trim();
    if (!text) {
      return res.status(400).json({ error: 'Please paste or type a message to check.' });
    }

    const voice = {
      verdict: 'text-only',
      score: null,
      synthetic: false,
      warning: 'Text message check (no audio).'
    };
    const pressure = checkPressure(text);
    const level = riskLevel(voice, pressure);

    const alert = await sendGuardianAlert(level, voice, pressure, text, null).catch(err => ({
      sent: false, reason: `Alert error: ${err.message}`
    }));

    res.json({
      level,
      checkedAt: new Date().toISOString(),
      voice,
      pressure: {
        count: pressure.count,
        categories: pressure.categories,
        matches: pressure.matches
      },
      transcript: text,
      sttError: null,
      alert: {
        sent: alert.sent,
        reason: alert.reason || null
      },
      disclaimer: 'These results are evidence, not proof.'
    });
  } catch (err) {
    console.error('[CheckText] Error:', err);
    res.status(500).json({ error: err.message || 'Text check failed' });
  }
});

// ─── WhatsApp Cloud API Webhook (for incoming messages) ───────────
app.get('/api/whatsapp/webhook', (req, res) => {
  const mode = req.query['hub.mode'];
  const token = req.query['hub.verify_token'];
  const challenge = req.query['hub.challenge'];
  if (mode === 'subscribe' && token === WA_VERIFY_TOKEN) {
    return res.status(200).send(challenge);
  }
  res.sendStatus(403);
});

app.post('/api/whatsapp/webhook', async (req, res) => {
  res.sendStatus(200);
  try {
    const value = req.body?.entry?.[0]?.changes?.[0]?.value;
    const messages = value?.messages || [];
    for (const msg of messages) {
      if (msg.type !== 'text' || !msg.text?.body) continue;
      const text = msg.text.body;
      const from = msg.from;
      const voice = { verdict: 'text-only', score: null, synthetic: false };
      const pressure = checkPressure(text);
      const level = riskLevel(voice, pressure);

      if (level !== 'calm') {
        const cats = Object.keys(pressure.categories).join(', ');
        if (WA_TOKEN && WA_PHONE_NUMBER_ID && from) {
          const reply = [
            `⚠️ *Hold On Alert (${level.toUpperCase()})*`,
            `This message looks suspicious (${cats}).`,
            `1. Do NOT pay money or share any OTP/code.`,
            `2. Call your family member on a saved number.`,
            `3. Wait 10 minutes before acting.`
          ].join('\n');
          await sendWhatsAppText(from, reply).catch(e => console.error('[WhatsApp] Reply failed:', e.message));
        }
        await sendGuardianAlert(level, voice, pressure, text, null);
      }
    }
  } catch (err) {
    console.error('[WhatsApp Webhook] Error:', err.message);
  }
});

// ─── Health check ────────────────────────────────────────────────
app.get('/api/health', (_, res) => {
  res.json({
    status: 'ok',
    aletheia: !!DETECTIF_API_KEY,
    stt: !!STT_KEY,
    telegram: !!(TG_TOKEN && TG_CHAT_ID),
    whatsapp: !!(GREEN_API_URL && GREEN_API_ID_INSTANCE && GREEN_API_TOKEN && WA_TO)
  });
});

// ─── Start ───────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🛡️  Hold On is running at http://localhost:${PORT}\n`);
  console.log(`   Aletheia API key: ${DETECTIF_API_KEY ? '✓ set' : '✗ MISSING'}`);
  console.log(`   STT key:         ${STT_KEY ? '✓ set' : '✗ missing (transcript disabled)'}`);
  console.log(`   Telegram:        ${TG_TOKEN && TG_CHAT_ID ? '✓ set' : '✗ missing (alerts disabled)'}`);
  console.log(`   WhatsApp:        ${GREEN_API_URL && GREEN_API_ID_INSTANCE && GREEN_API_TOKEN && WA_TO ? '✓ set' : '✗ missing (alerts disabled)'}`);
  console.log(`   ffmpeg:          ${FFMPEG_PATH}\n`);
});
// Log health info on startup
const healthInfo = {
  status: 'ok',
  aletheia: !!DETECTIF_API_KEY,
  stt: !!STT_KEY,
  telegram: !!(TG_TOKEN && TG_CHAT_ID),
  whatsapp: !!(GREEN_API_URL && GREEN_API_ID_INSTANCE && GREEN_API_TOKEN && WA_TO)
};
console.log('Health check:', JSON.stringify(healthInfo));
