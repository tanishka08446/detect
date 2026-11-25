# 🛡️ Hold On

**Scam call defence: deepfake voice detection + pressure-tactic analysis + guardian alerts**

> *"A real voice is not proof of safety. A fake voice is not proof of a scam."*

Built for the **CMRIT × DetectifAI Hackathon** – Track 01: Scam Shield.

---

## What it does

Scam calls work by creating **urgency** and **isolation** ("Don't tell anyone, send money now"). Hold On breaks both by combining **three signals**:

1. **Aletheia Voice Check** – Uses DetectifAI's Aletheia API to detect if the voice is AI-generated/cloned.
2. **Pressure-Tactic Check** – Transcribes the audio and flags scam tactics: urgency, secrecy, money/UPI/OTP demands, and authority threats (police, CBI, "digital arrest"). Matched phrases are highlighted in the transcript.
3. **Guardian Alert** – Sends a Telegram message with the risk level, evidence summary, and the audio clip itself to a trusted family member. The victim is no longer alone.

The app speaks the results aloud in **English, Hindi, or Kannada** with a 3-step safety checklist:
1. Don't pay or share any code.
2. Hang up and call your relative on the number you already have saved.
3. Wait 10 minutes and talk to family.

### Why voice detection alone isn't enough

Many scams, like "digital arrest" calls, use **real human voices**. A deepfake detector alone would say "human voice detected" — which a victim might interpret as "safe". That's dangerous.

Hold On's risk logic:
- **HIGH** = synthetic voice AND ≥2 pressure categories
- **WARN** = synthetic voice OR ≥2 pressure categories
- **CALM** = no strong warning signs (but this is **not** a guarantee of safety)

The app **never says "safe"**.

---

## Setup

### Prerequisites
- **Node.js 18+** ([download](https://nodejs.org/))
- **ffmpeg** – Either install system-wide ([download](https://ffmpeg.org/download.html)) or the app will use the bundled `ffmpeg-static` npm package as a fallback.

### Install

```bash
cd hold-on
npm install
```

### Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `DETECTIF_API_KEY` | ✅ Yes | DetectifAI Aletheia API key |
| `STT_URL` | No | Whisper-compatible STT endpoint (default: Groq) |
| `STT_MODEL` | No | STT model name (default: `whisper-large-v3`) |
| `STT_KEY` | No | STT API key (transcript disabled if missing) |
| `TG_TOKEN` | No | Telegram bot token from @BotFather |
| `TG_CHAT_ID` | No | Telegram chat ID for guardian alerts |
| `PORT` | No | Server port (default: 3000) |

### Run

```bash
npm start
```

Open **http://localhost:3000** in your browser.

---

## Telegram Guardian Alert setup

1. **Create a bot**: Message [@BotFather](https://t.me/BotFather) on Telegram → `/newbot` → follow prompts → copy the token.
2. **Get your chat ID**:
   - Send a message to your new bot.
   - Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
   - Find `"chat": {"id": 123456789}` in the response — that's your `TG_CHAT_ID`.
3. Add both to your `.env` file.

When Hold On detects a "warn" or "high" risk, it sends:
- A text message with the risk level, voice verdict, pressure categories, and transcript excerpt.
- The audio clip as a document attachment.

---

## Running tests

Place audio clips in the following directories:
```
test/clips/cloned/     – cloned/synthetic voice clips
test/clips/real_scam/  – real human voice reading scam scripts
test/clips/normal/     – normal family voice notes
```

With the server running:
```bash
npm test
```

This prints a results table and generates `results.md`.

**Expected results:**
| Group | Expected Level |
|-------|---------------|
| cloned | warn or high |
| real_scam | warn or high |
| normal | calm |

---

## Demo script

### Clip 1: Cloned voice scam
> *"Mom, I'm in trouble, send 20,000 rupees on UPI now, don't tell dad."*

**Expected**: Aletheia flags synthetic → pressure check finds urgency + money + secrecy → **HIGH** → Guardian alert sent.

### Clip 2: Real human digital-arrest scam
> *"This is CBI. A case is registered against you. Transfer money immediately or you will be arrested."*

**Expected**: Aletheia says human → pressure check finds threat + urgency + money → **WARN**. This shows what voice detectors alone miss.

### Clip 3: Normal family voice note
> *"Hey, just checking in. Call me when you're free!"*

**Expected**: Human voice, no pressure tactics → **CALM**, no alert. It doesn't cry wolf.

---

## Limitations

> [!IMPORTANT]
> **This is a hackathon prototype, not production software.**

- **Keyword heuristic**: The pressure-tactic check uses simple keyword matching, not NLP. It will have false positives (flagging innocent phrases) and false negatives (missing novel scam scripts).
- **STT accuracy**: Whisper's accuracy varies significantly for Hindi and Kannada, especially with background noise or code-switching. Transcription errors directly affect pressure detection.
- **"Calm" ≠ "Safe"**: The calm state means no strong warning signs were detected. It is NOT proof that the call is safe. The app always shows this disclaimer.
- **Shared API credits**: Aletheia credits are limited and shared across hackathon teams. The app caches results by audio hash to avoid re-spending credits.
- **Aletheia response mapper**: The mapper (`readVoice()`) is designed to be flexible with unknown response field names. If using a new API version, check the console logs for the raw response and update the mapper accordingly.
- **Browser TTS**: Speech synthesis quality and language availability varies by browser and OS. Hindi and Kannada voices may not be available on all systems.
- **Not a replacement for judgment**: This tool provides evidence to help users pause and think. It cannot guarantee detection of all scams.

---

## Privacy

- Uploaded and converted temp files are **deleted after each request**.
- No audio or transcripts are stored on disk beyond the request.
- The only intentional data transmission is the Guardian alert (to the user's own Telegram contact).
- All test audio uses **consenting voices** (teammates only).

---

## Tech stack

- Node.js + Express + multer
- DetectifAI Aletheia API (deepfake detection)
- Groq Whisper API (speech-to-text)
- Telegram Bot API (guardian alerts)
- ffmpeg / ffmpeg-static (audio conversion)
- Vanilla HTML/CSS/JS (no frameworks, no CDN)

---

## File structure

```
├── server.js              # Express backend
├── public/index.html      # Frontend (single page)
├── test/run-tests.js      # Test harness
├── test/clips/            # Test audio clips (not committed)
├── package.json
├── .env.example           # Environment template
├── .gitignore
└── README.md
```

---

*Hold On – CMRIT × DetectifAI Hackathon, Track 01: Scam Shield*
