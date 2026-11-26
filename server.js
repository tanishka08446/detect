// server.js – Hold On backend
// Combines DetectifAI Aletheia (deepfake voice detection) + pressure-tactic
// keyword heuristic + Telegram guardian alerts.

import 'dotenv/config';
import express from 'express';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import multer from 'multer';
import { createHash, randomUUID } from 'crypto';
import { readFile, unlink, mkdir, stat, appendFile } from 'fs/promises';
import { existsSync } from 'fs';
import { join, extname } from 'path';
import { exec, execFile } from 'child_process';
import { promisify } from 'util';
import { fileURLToPath } from 'url';
import FormData from 'form-data';
import fetch from 'node-fetch';

const execAsync = promisify(exec);
const execFileAsync = promisify(execFile);
const __dirname = fileURLToPath(new URL('.', import.meta.url));

// ─── Environment validation ───────────────────────────────────────
const REQUIRED_ENV = ['DETECTIF_API_KEY'];
const missingEnv = REQUIRED_ENV.filter(k => !process.env[k]);
if (missingEnv.length) {
  console.error(`\n❌  Missing required environment variable(s): ${missingEnv.join(', ')}`);
  console.error('    Please set them in your .env file and restart.\n');
  process.exit(1);
}

// ─── Config ───────────────────────────────────────────────────────
const PORT = 3003;
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

// Security headers
app.use(helmet({ contentSecurityPolicy: false }));

// Rate limiting: max 20 analysis requests per 15 minutes per IP
const analysisLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 20,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests. Please wait a few minutes and try again.' }
});

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

// Apply rate limiter to analysis endpoints
app.use('/api/analyze', analysisLimiter);
app.use('/api/check-text', analysisLimiter);

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
  if (lang && lang !== 'auto') form.append('language', lang);

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

// ─── Pressure-tactic keyword heuristic (loaded from dataset) ─────
// NOTE: This is a simple keyword/phrase heuristic, NOT NLP or ML.
// It will have false positives and negatives. See README for limitations.
const keywordDataset = JSON.parse(
  await readFile(join(__dirname, 'data', 'scam-keywords.json'), 'utf8')
);
const PRESSURE_KEYWORDS = keywordDataset.pressureKeywords;

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
// "critical" = ≥3 coercion vectors OR (critical phrase + financial demand) OR (synthetic voice + ≥3 vectors)
// "high" = synthetic voice AND (≥2 pressure categories OR critical phrase)
// "warn" = synthetic voice OR ≥2 pressure categories OR critical phrase
// "safe" = no pressure vectors and not synthetic voice
// "calm" = otherwise
const CRITICAL_PHRASES = new Set(
  keywordDataset.criticalPhrases.map(p => p.toLowerCase())
);
const FINANCIAL_PHRASES = new Set([
  'upi', 'bank', 'payment', 'transfer', 'money', 'financial', 'account'
]);

function riskLevel(voice, pressure) {
  const isSynthetic = voice.synthetic;
  const pressureCount = pressure.count;
  const hasCritical = Object.values(pressure.matches || {})
    .flat()
    .some(p => CRITICAL_PHRASES.has(p.toLowerCase()));
  const hasFinancial = Object.values(pressure.matches || {})
    .flat()
    .some(p => FINANCIAL_PHRASES.has(p.toLowerCase()));

  // SAFE: no pressure vectors and not synthetic voice
  if (!isSynthetic && pressureCount === 0) return 'safe';

  // CRITICAL tier detection
  if (pressureCount >= 3 || (hasCritical && hasFinancial) || (isSynthetic && pressureCount >= 3)) {
    return 'critical';
  }

  // HIGH tier detection
  if (isSynthetic && (pressureCount >= 2 || hasCritical)) return 'high';

  // WARN tier detection
  if (isSynthetic || pressureCount >= 2 || hasCritical) return 'warn';

  // CALM as fallback
  return 'calm';
}

// ─── Guardian Alerts (Telegram + WhatsApp) ──────────────────────
function buildAlertMessage(level, voice, pressure, transcript, safeWord) {
  const safeWordCheck = `\n🔐 Family Safe-Word Check: Ask caller to confirm code word "${safeWord || 'MANGO-7'}" before taking action.`;
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
    `⚠️ Please check on your family member. This is an automated alert from Hold On.${safeWordCheck}`
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
    return { channel: 'Telegram', sent: false, reason: err.message, networkError: isNetworkError(err) };
  }
}

function isNetworkError(err) {
  const msg = String(err?.message || err || '');
  const code = String(err?.code || err?.cause?.code || '');
  return (
    ['ENOTFOUND', 'ECONNREFUSED', 'ETIMEDOUT', 'ECONNRESET', 'EAI_AGAIN'].includes(code) ||
    /ENOTFOUND|ECONNREFUSED|ETIMEDOUT|ECONNRESET|EAI_AGAIN|fetch failed|unreachable|network/i.test(msg)
  );
}

// ─── Offline Retry Queue (WhatsApp & Telegram) ───────────────────
const alertRetryQueue = [];

const ADB_CANDIDATES = [
  process.env.ADB_PATH,
  'adb',
  process.env.LOCALAPPDATA ? join(process.env.LOCALAPPDATA, 'Android', 'Sdk', 'platform-tools', 'adb.exe') : null,
  'C:\\platform-tools\\adb.exe'
].filter(Boolean);

let RESOLVED_ADB = 'adb';
for (const candidate of ADB_CANDIDATES) {
  if (candidate === 'adb' || existsSync(candidate)) {
    try {
      await execAsync(`"${candidate}" version`, { timeout: 2000 });
      RESOLVED_ADB = candidate;
      break;
    } catch { /* try next */ }
  }
}

function formatSmsPhone(raw) {
  if (!raw) return null;
  let cleaned = String(raw).replace(/@c\.us$/i, '').replace(/@s\.whatsapp\.net$/i, '').replace(/[\s\-()]/g, '');
  if (!cleaned) return null;
  if (!cleaned.startsWith('+')) {
    if (/^\d{10}$/.test(cleaned)) cleaned = '+91' + cleaned;
    else cleaned = '+' + cleaned;
  }
  return /^\+\d{7,15}$/.test(cleaned) ? cleaned : null;
}

async function checkAdbDevice() {
  try {
    const { stdout } = await execAsync(`"${RESOLVED_ADB}" devices`, { timeout: 3000 });
    const lines = stdout.split(/\r?\n/).slice(1);
    return lines.some(line => /\bdevice\b/.test(line.trim()));
  } catch {
    return false;
  }
}

async function sendSmsAlert(message) {
  const phone = formatSmsPhone(process.env.WA_TO || WA_TO);
  if (!phone) {
    console.warn('[SMS] Cannot send SMS: WA_TO is missing or invalid.');
    return { channel: 'SMS', sent: false, reason: 'WA_TO not configured' };
  }

  const hasDevice = await checkAdbDevice();
  if (!hasDevice) {
    console.warn('[SMS] No Android device detected via adb devices – skipping ADB SMS.');
    return { channel: 'SMS', sent: false, reason: 'Offline Cellular Standby Ready' };
  }

  try {
    // Strip markdown asterisks and collapse newlines/spaces for clean SMS parcel string
    const cleanBody = String(message)
      .replace(/\*/g, '')
      .replace(/\r?\n+/g, ' | ')
      .slice(0, 320);

    // Escape quotes, backslashes, and shell metacharacters for adb shell service call isms (Android 11-15+)
    const escapedBody = cleanBody
      .replace(/\\/g, '\\\\')
      .replace(/"/g, '\\"')
      .replace(/\$/g, '\\$')
      .replace(/`/g, '\\`');

    const cmd = `"${RESOLVED_ADB}" shell service call isms 5 i32 0 s16 "com.android.shell" s16 "null" s16 "${phone}" s16 "null" s16 "${escapedBody}" s16 "null" s16 "null" i32 0 i64 0`;
    const { stdout, stderr } = await execAsync(cmd, { timeout: 10_000 });

    if (stderr && /error|exception|not found/i.test(stderr)) {
      console.error('[SMS] ADB SMS failed:', stderr.trim());
      return { channel: 'SMS', sent: false, reason: stderr.trim() };
    }

    console.log(`[SMS] ADB SMS sent to ${phone}:`, stdout.trim() || 'OK');
    return { channel: 'SMS', sent: true, reason: null };
  } catch (err) {
    console.error('[SMS] ADB SMS error:', err.message);
    return { channel: 'SMS', sent: false, reason: err.message };
  }
}

async function sendWhatsAppText(to, body) {
  // GREEN-API sendMessage endpoint
  const chatTarget = String(to || WA_TO).replace(/@c\.us$/i, '');
  const endpoint = `${GREEN_API_URL}/waInstance${GREEN_API_ID_INSTANCE}/sendMessage/${GREEN_API_TOKEN}`;
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      chatId: `${chatTarget}@c.us`,
      message: body
    })
  });
  if (!res.ok) {
    const errText = await res.text().catch(() => '');
    throw new Error(`GREEN-API HTTP ${res.status}: ${errText.slice(0, 150)}`);
  }
  return res.json().catch(() => ({}));
}

async function sendWhatsAppAlert(level, alertText, wavPath) {
  if (!GREEN_API_URL || !GREEN_API_ID_INSTANCE || !GREEN_API_TOKEN || !WA_TO) {
    return { channel: 'WhatsApp', sent: false, reason: 'not configured' };
  }

  try {
    const data = await sendWhatsAppText(WA_TO, alertText);

    if (wavPath && existsSync(wavPath)) {
      const audioBuffer = await readFile(wavPath);
      const form = new FormData();
      const chatTarget = String(WA_TO).replace(/@c\.us$/i, '');
      form.append('chatId', `${chatTarget}@c.us`);
      form.append('file', audioBuffer, { filename: 'suspicious_clip.wav', contentType: 'audio/wav' });

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
    return { channel: 'WhatsApp', sent: true, reason: null, idMessage: data?.idMessage || null };
  } catch (err) {
    console.error('[WhatsApp] Alert error:', err.message);
    return { channel: 'WhatsApp', sent: false, reason: err.message, networkError: isNetworkError(err) };
  }
}

async function sendGuardianAlert(level, voice, pressure, transcript, wavPath, safeWord) {
  if (level === 'calm') {
    return { sent: false, reason: 'Level is calm – no alert needed' };
  }

  const alertText = buildAlertMessage(level, voice, pressure, transcript, safeWord);
  const [tgResult, waResult, smsResult] = await Promise.all([
    sendTelegramAlert(level, alertText, wavPath),
    sendWhatsAppAlert(level, alertText, wavPath),
    sendSmsAlert(alertText)
  ]);

  // Queue failed WhatsApp or Telegram alerts when offline
  let queuedCount = 0;
  if (!waResult.sent && waResult.networkError) {
    alertRetryQueue.push({
      id: randomUUID(),
      channel: 'whatsapp',
      level,
      alertText,
      queuedAt: new Date().toISOString()
    });
    queuedCount++;
    console.log(`[RetryQueue] Queued WhatsApp alert (queue size: ${alertRetryQueue.length})`);
  }
  if (!tgResult.sent && tgResult.networkError) {
    alertRetryQueue.push({
      id: randomUUID(),
      channel: 'telegram',
      level,
      alertText,
      queuedAt: new Date().toISOString()
    });
    queuedCount++;
    console.log(`[RetryQueue] Queued Telegram alert (queue size: ${alertRetryQueue.length})`);
  }

  const results = [tgResult, waResult, smsResult];
  const sentChannels = results.filter(r => r.sent).map(r => r.channel);
  const failed = results.filter(r => !r.sent && r.reason !== 'not configured' && r.reason !== 'Offline Cellular Standby Ready');
  const standby = results.filter(r => r.reason === 'Offline Cellular Standby Ready');

  if (sentChannels.length > 0) {
    const queueNote = queuedCount > 0 ? `; ${queuedCount} queued for online retry` : '';
    const standbyNote = standby.length > 0 ? ' (SMS: Offline Cellular Standby Ready)' : '';
    const note = failed.length > 0
      ? `Sent via ${sentChannels.join(' & ')} (${failed.map(f => `${f.channel}: ${f.reason}`).join('; ')}${queueNote})${standbyNote}`
      : `Sent via ${sentChannels.join(' & ')}${queueNote}${standbyNote}`;
    return { sent: true, reason: note };
  }

  if (queuedCount > 0) {
    return {
      sent: false,
      reason: `Offline – ${queuedCount} alert(s) queued for retry (${failed.map(f => `${f.channel}: ${f.reason}`).join('; ')})`
    };
  }

  return {
    sent: false,
    reason: failed.map(f => `${f.channel}: ${f.reason}`).join('; ') || 'Alert could not be sent'
  };
}

// Background retry loop every 10 seconds to flush alertRetryQueue once online
setInterval(async () => {
  if (alertRetryQueue.length === 0) return;

  for (let i = alertRetryQueue.length - 1; i >= 0; i--) {
    const item = alertRetryQueue[i];
    try {
      if (item.channel === 'whatsapp') {
        const res = await sendWhatsAppText(WA_TO, item.alertText);
        if (res && res.idMessage) {
          console.log(`[RetryQueue] Flushed WhatsApp alert ${item.id} (idMessage: ${res.idMessage})`);
          alertRetryQueue.splice(i, 1);
        }
      } else if (item.channel === 'telegram') {
        const tgRes = await sendTelegramAlert(item.level, item.alertText, null);
        if (tgRes.sent) {
          console.log(`[RetryQueue] Flushed Telegram alert ${item.id}`);
          alertRetryQueue.splice(i, 1);
        }
      }
    } catch (err) {
      if (!isNetworkError(err)) {
        console.error(`[RetryQueue] Permanent error for ${item.channel} ${item.id}:`, err.message);
      }
    }
  }
}, 10_000);

// ─── Cleanup helper ──────────────────────────────────────────────
async function cleanup(...paths) {
  for (const p of paths) {
    try { if (p) await unlink(p); } catch { /* already gone */ }
  }
}

// ─── Local rule-based offline scam detector ──────────────────────
const OFFLINE_RED_FLAGS = [
  'otp', 'upi', 'pin', 'blocked', 'suspended', 'fraud department',
  'kyc', 'bank', 'urgent', 'immediately', 'transfer', 'police', 'arrest'
];

function detectLocalScam(text) {
  if (!text) return { isScam: false, matchedFlags: [], pressure: { categories: {}, count: 0, matches: {} } };
  const lower = text.toLowerCase();
  const matchedFlags = OFFLINE_RED_FLAGS.filter(flag => {
    const pattern = new RegExp(`\\b${escapeRegex(flag)}\\b`, 'i');
    return pattern.test(text) || lower.includes(flag);
  });
  const pressure = checkPressure(text);
  const isScam = matchedFlags.length >= 2 || pressure.count >= 2;
  return { isScam, matchedFlags, pressure };
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
    const fallbackText = (req.body?.text || req.body?.transcript || '').trim();

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

    // 3. Run Aletheia + STT in parallel with offline network fallback
    let voice;
    let transcript = null;
    let sttError = null;
    let offlineMode = false;
    let isScam = false;
    let pressure;
    let level;

    try {
      const [aletheiaRes, sttResult] = await Promise.all([
        callAletheia(wavPath),
        transcribe(wavPath, lang)
      ]);
      voice = aletheiaRes;
      transcript = sttResult.transcript;
      sttError = sttResult.error;

      if (sttError && isNetworkError(sttError)) {
        offlineMode = true;
      }

      pressure = checkPressure(transcript || fallbackText || '');
      const localCheck = detectLocalScam(transcript || fallbackText || '');
      level = riskLevel(voice, pressure);
      isScam = level !== 'calm' || localCheck.isScam;
    } catch (cloudErr) {
      if (!isNetworkError(cloudErr)) {
        throw cloudErr;
      }
      console.warn('[Analyze] Cloud API unreachable (offline):', cloudErr.message);
      offlineMode = true;

      const availableText = fallbackText || '';
      const localCheck = detectLocalScam(availableText);
      pressure = localCheck.pressure;
      if (localCheck.matchedFlags.length > 0 && !pressure.categories.offline_flags) {
        pressure.matches.offline_flags = localCheck.matchedFlags;
      }
      isScam = localCheck.isScam;
      level = isScam ? 'high' : 'calm';
      transcript = availableText || null;
      sttError = 'Offline mode: cloud speech-to-text unavailable.';
      voice = {
        verdict: 'unverified',
        score: null,
        synthetic: false,
        warning: 'Offline mode: cloud deepfake voice check unavailable. Using local rule-based scam detector.'
      };
    }

    // 4. Guardian alert (skip if safe)
    if (level === 'safe') {
      // Immediate safe response, no alerts sent
      return res.json({
        level,
        isScam: false,
        offlineMode,
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
        alert: { sent: false, reason: null },
        disclaimer: 'These results are evidence, not proof.',
        risk: 'safe',
        score: 5,
        status: 'SAFE / VERIFIED'
      });
    }
    const safeWord = req.body?.safeWord || 'MANGO-7';
    const alert = await sendGuardianAlert(level, voice, pressure, transcript, wavPath, safeWord).catch(err => ({
      sent: false, reason: `Alert error: ${err.message}`
    }));

    // 5. Log result to results.md
    const logEntry = [
      `## ${new Date().toLocaleString()} — ${level.toUpperCase()}${offlineMode ? ' (OFFLINE)' : ''}`,
      `- **Voice:** ${voice.verdict}${voice.score !== null ? ` (score: ${voice.score})` : ''}`,
      `- **Pressure:** ${Object.keys(pressure.categories).join(', ') || 'none'}`,
      `- **Transcript:** ${transcript ? transcript.slice(0, 200) : 'unavailable'}`,
      ''
    ].join('\n');
    appendFile(join(__dirname, 'results.md'), logEntry).catch(() => {});

    // 6. Respond
    res.json({
      level,
      isScam,
      offlineMode,
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
    const localCheck = detectLocalScam(text);
    const pressure = localCheck.pressure;
    let level = riskLevel(voice, pressure);
      if (level === 'safe') {
        // Immediate safe response, no alerts sent
        return res.json({
          risk: 'safe',
          score: 5,
          status: 'SAFE / VERIFIED',
          headline: 'No Scam Indicators Detected'
        });
      }
    if (localCheck.isScam && level === 'calm') {
      level = 'warn';
    }
    const isScam = level !== 'calm' || localCheck.isScam;

    const safeWord = req.body?.safeWord || 'MANGO-7';
      const alert = await sendGuardianAlert(level, voice, pressure, text, safeWord).catch(err => ({
      sent: false, reason: `Alert error: ${err.message}`
    }));

    res.json({
      level,
      isScam,
      offlineMode: false,
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
app.get('/api/health', async (_, res) => {
  const adbConnected = await checkAdbDevice();
  const smsConfigured = !!formatSmsPhone(process.env.WA_TO || WA_TO);
  res.json({
    status: 'ok',
    aletheia: !!DETECTIF_API_KEY,
    stt: !!STT_KEY,
    telegram: !!(TG_TOKEN && TG_CHAT_ID),
    whatsapp: !!(GREEN_API_URL && GREEN_API_ID_INSTANCE && GREEN_API_TOKEN && WA_TO),
    sms: adbConnected || smsConfigured,
    adbConnected,
    queuedAlerts: alertRetryQueue.length
  });
});

// ─── Start ───────────────────────────────────────────────────────
const startupAdbConnected = await checkAdbDevice();
const startupSmsReady = startupAdbConnected || !!formatSmsPhone(process.env.WA_TO || WA_TO);

const server = app.listen(PORT, () => {
  console.log(`\n🛡️  Hold On is running at http://localhost:${PORT}\n`);
  console.log(`   Aletheia API key: ${DETECTIF_API_KEY ? '✓ set' : '✗ MISSING'}`);
  console.log(`   STT key:         ${STT_KEY ? '✓ set' : '✗ missing (transcript disabled)'}`);
  console.log(`   Telegram:        ${TG_TOKEN && TG_CHAT_ID ? '✓ set' : '✗ missing (alerts disabled)'}`);
  console.log(`   WhatsApp:        ${GREEN_API_URL && GREEN_API_ID_INSTANCE && GREEN_API_TOKEN && WA_TO ? '✓ set' : '✗ missing (alerts disabled)'}`);
  console.log(`   SMS (ADB):       ${startupAdbConnected ? '✓ Android device connected' : (startupSmsReady ? '✓ enabled (no ADB phone plugged in)' : '✗ missing WA_TO')}`);
  console.log(`   ffmpeg:          ${FFMPEG_PATH}\n`);
});

// ─── Graceful shutdown ───────────────────────────────────────────
function shutdown(signal) {
  console.log(`\n[Hold On] Received ${signal}. Shutting down gracefully…`);
  server.close(() => {
    console.log('[Hold On] Server closed. Goodbye!');
    process.exit(0);
  });
  setTimeout(() => { console.error('[Hold On] Force exit after timeout.'); process.exit(1); }, 10_000);
}
process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT',  () => shutdown('SIGINT'));
// Log health info on startup
const healthInfo = {
  status: 'ok',
  aletheia: !!DETECTIF_API_KEY,
  stt: !!STT_KEY,
  telegram: !!(TG_TOKEN && TG_CHAT_ID),
  whatsapp: !!(GREEN_API_URL && GREEN_API_ID_INSTANCE && GREEN_API_TOKEN && WA_TO),
  sms: startupSmsReady,
  adbConnected: startupAdbConnected,
  queuedAlerts: alertRetryQueue.length
};
console.log('Health check:', JSON.stringify(healthInfo));
