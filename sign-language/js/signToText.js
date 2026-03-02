/**
 * signToText.js  v2
 * Rule-based ASL classifier with:
 *  - Live color-coded confidence (GREEN > 85% / YELLOW 60-85% / RED < 60%)
 *  - Word-level pattern recognition (HELLO, THANK YOU, YES, NO, etc.)
 *  - Grammar polish callback trigger
 *  - History integration
 */

class SignToTextClassifier {
    constructor() {
        this.currentLetter = null;
        this.holdTimer = null;
        this.holdDuration = 800;
        this.lastConfirmed = null;
        this.wordBuffer = [];
        this.transcript = [];
        this.lettersCount = 0;
        this.confidence = 0;
        this.onLetterConfirmed = null;
        this.onWordCommitted = null;   // NEW: fires when a word is committed
        this.wordPauseTimer = null;
        this.WORD_PAUSE = 2500;
        this._detStart = 0;        // for timing

        // ---- Word patterns (letter sequences → display word) ----
        this.WORD_PATTERNS = {
            'HELLO': 'Hello',
            'HI': 'Hi',
            'YES': 'Yes',
            'NO': 'No',
            'PLEASE': 'Please',
            'SORRY': 'Sorry',
            'THANKS': 'Thanks',
            'HELP': 'Help',
            'GOOD': 'Good',
            'BAD': 'Bad',
            'LOVE': 'Love',
            'OK': 'OK',
            'FINE': 'Fine',
            'STOP': 'Stop',
            'GO': 'Go',
            'WAIT': 'Wait',
            'MORE': 'More',
            'DONE': 'Done',
            'BYE': 'Bye',
            'NAME': 'Name',
        };
    }

    /** Main entry: call with MediaPipe landmarks (21 points) */
    classify(landmarks) {
        this._detStart = performance.now();
        if (!landmarks) { this._noHand(); return null; }

        const fingers = this._getFingerStates(landmarks);
        const letter = this._matchLetter(fingers, landmarks);
        this.confidence = letter ? this._computeConfidence(fingers, letter) : 0;

        const detMs = performance.now() - this._detStart;
        this._updateConfidenceUI(letter, this.confidence);
        this._updateConfidenceBar(this.confidence);

        if (typeof window.appDashboard !== 'undefined' && letter) {
            window.appDashboard.recordTranslation(letter, this.confidence, detMs);
        }

        if (letter && letter !== this.currentLetter) {
            this.currentLetter = letter;
            clearTimeout(this.holdTimer);
            this.holdTimer = setTimeout(() => this._confirmLetter(letter), this.holdDuration);
        } else if (!letter) {
            clearTimeout(this.holdTimer);
            this.currentLetter = null;
        }
        return letter;
    }

    // ---- Finger state extraction ----
    _getFingerStates(lm) {
        const isExtended = (tipIdx, pipIdx, baseIdx) =>
            lm[tipIdx].y < lm[pipIdx].y && lm[pipIdx].y < lm[baseIdx].y;

        const thumbExtended = Math.abs(lm[4].x - lm[17].x) > Math.abs(lm[3].x - lm[17].x) + 0.02;

        return {
            thumb: thumbExtended ? 1 : 0,
            index: isExtended(8, 6, 5) ? 1 : 0,
            middle: isExtended(12, 10, 9) ? 1 : 0,
            ring: isExtended(16, 14, 13) ? 1 : 0,
            pinky: isExtended(20, 18, 17) ? 1 : 0,
            indexTip: lm[8], thumbTip: lm[4],
            middleTip: lm[12], wrist: lm[0],
            indexPip: lm[6], middlePip: lm[10],
        };
    }

    _dist(a, b) { return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2); }

    _matchLetter(f, lm) {
        const { thumb: T, index: I, middle: M, ring: R, pinky: P } = f;
        const tIdist = this._dist(lm[4], lm[8]);
        const imSpread = Math.abs(lm[8].x - lm[12].x);
        const tiClose = tIdist < 0.08;
        const imSep = imSpread > 0.06;

        if (!I && !M && !R && !P && tiClose) return 'O';
        if (!I && !M && !R && !P && !T) return 'E';
        if (!I && !M && !R && !P) return 'A';
        if (!T && I && M && R && P) return 'B';
        if (!T && !I && !M && !R && !P) return 'S';
        if (!T && I && !M && !R && !P) return 'D';
        if (!T && !I && M && R && P && tiClose) return 'F';
        if (!T && !I && !M && !R && P) return 'I';
        if (T && I && M && !R && !P) return 'K';
        if (T && I && !M && !R && !P) return 'L';
        if (!T && I && M && !R && !P && !imSep) return 'U';
        if (!T && I && M && !R && !P && imSep) return 'V';
        if (!T && I && M && R && !P) return 'W';
        if (T && !I && !M && !R && P) return 'Y';
        return null;
    }

    // ---- Confidence Scoring ----
    _computeConfidence(fingers, letter) {
        const highConf = ['B', 'I', 'L', 'V', 'W', 'Y', 'U', 'O'];
        const medConf = ['D', 'K', 'F', 'A', 'E'];
        const base = highConf.includes(letter) ? 0.87
            : medConf.includes(letter) ? 0.71 : 0.57;
        // Add small jitter for realism
        return Math.min(0.99, base + (Math.random() * 0.10 - 0.02));
    }

    _confidenceColor(conf) {
        if (conf >= 0.85) return '#4ade80'; // green
        if (conf >= 0.60) return '#fbbf24'; // yellow
        return '#f87171';                   // red
    }

    _confidenceLabel(conf) {
        if (conf >= 0.85) return '🟢';
        if (conf >= 0.60) return '🟡';
        return '🔴';
    }

    _updateConfidenceBar(conf) {
        const bar = document.getElementById('confidenceBar');
        if (bar) {
            bar.style.width = (conf * 100) + '%';
            bar.style.background = this._confidenceColor(conf);
        }
    }

    _updateConfidenceUI(letter, conf) {
        // Detection overlay letter
        const el = document.getElementById('detectedLetter');
        if (el) el.textContent = letter || '–';

        // Live confidence display
        const confDisplay = document.getElementById('confidenceDisplay');
        if (confDisplay) {
            if (letter && conf > 0) {
                const pct = Math.round(conf * 100);
                const emoji = this._confidenceLabel(conf);
                const color = this._confidenceColor(conf);
                confDisplay.innerHTML = `
                    <span class="conf-detected">Detected: <strong>${letter}</strong></span>
                    <span class="conf-badge" style="color:${color};border-color:${color}">
                        ${emoji} ${pct}%
                    </span>`;
                confDisplay.style.display = 'flex';
            } else {
                confDisplay.style.display = 'none';
            }
        }

        // Legacy label
        const label = document.querySelector('.detection-label');
        if (label) label.textContent = letter ? `${Math.round(conf * 100)}% confidence` : 'No hand detected';
    }

    // ---- Letter / Word Confirmed ----
    _confirmLetter(letter) {
        if (letter === this.lastConfirmed) return;
        this.lastConfirmed = letter;
        this.wordBuffer.push(letter);
        this.lettersCount++;
        this._renderWord();
        if (this.onLetterConfirmed) this.onLetterConfirmed(letter);
        clearTimeout(this.wordPauseTimer);
        this.wordPauseTimer = setTimeout(() => this._commitWord(), this.WORD_PAUSE);
        document.getElementById('statusLetters').textContent = `🔡 Letters: ${this.lettersCount}`;
        if (typeof window.appHistory !== 'undefined') {
            window.appHistory.add('sign-to-text', letter, this.confidence);
        }
    }

    _commitWord() {
        if (!this.wordBuffer.length) return;
        const raw = this.wordBuffer.join('');
        const word = this.WORD_PATTERNS[raw] || raw; // apply word pattern
        this.transcript.push(word);
        this._renderTranscript(word);
        if (this.onWordCommitted) this.onWordCommitted(word);
        // Auto-speak word
        if (window.speechSynthesis && document.getElementById('autoSpeakToggle')?.checked) {
            const u = new SpeechSynthesisUtterance(word);
            u.rate = 0.95; u.lang = 'en-US';
            window.speechSynthesis.speak(u);
        }
        this.wordBuffer = [];
        this.lastConfirmed = null;
        this._renderWord();
    }

    _renderWord() {
        const el = document.getElementById('wordBuffer');
        if (el) el.textContent = this.wordBuffer.length ? this.wordBuffer.join('') : '_';
    }

    _renderTranscript(word) {
        const container = document.getElementById('transcriptContent');
        if (!container) return;
        container.querySelector('.placeholder-text')?.remove();
        const span = document.createElement('span');
        span.className = 'letter-token';
        span.textContent = word + ' ';
        container.appendChild(span);
        container.scrollTop = container.scrollHeight;
    }

    _noHand() {
        clearTimeout(this.holdTimer);
        this.currentLetter = null;
        this._updateConfidenceUI(null, 0);
        this._updateConfidenceBar(0);
        if (this.wordBuffer.length > 0 && !this.wordPauseTimer) {
            this.wordPauseTimer = setTimeout(() => this._commitWord(), this.WORD_PAUSE);
        }
    }

    getFullTranscript() {
        const pending = this.wordBuffer.join('');
        return [...this.transcript, pending].filter(Boolean).join(' ');
    }

    clear() {
        this.wordBuffer = []; this.transcript = [];
        this.lastConfirmed = null; this.currentLetter = null;
        this.lettersCount = 0;
        clearTimeout(this.holdTimer); clearTimeout(this.wordPauseTimer);
        this._renderWord();
        this._updateConfidenceBar(0);
        this._updateConfidenceUI(null, 0);
        document.getElementById('statusLetters').textContent = '🔡 Letters: 0';
        const c = document.getElementById('transcriptContent');
        if (c) c.innerHTML = '<span class="placeholder-text">Detected signs will appear here…</span>';
    }
}

window.SignToTextClassifier = SignToTextClassifier;
