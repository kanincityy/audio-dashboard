import { useState, useEffect } from 'react'
import './App.css'

const MAIN_KEY = {
  format: "sample_rate_khz",
  transcript: "overall_confidence",
  levels: "peak_dbfs",
  clipping: "clipped_sample_pct",
  bandwidth: "effective_cutoff_khz",
  vad: "segment_count",
  snr: "snr_db",
  rt60: "rt60_s",
  density: "speech_ratio",
  truncation: "word_count",
  tempo: "wpm",
  overlap: "overlap_seconds",
  speakers: "speaker_count",
  f0: "f0_median_hz",
};
const UNIT = {
  format: "kHz",
  levels: "dBFS",
  clipping: "%",
  bandwidth: "kHz",
  vad: "segments",
  snr: "dB",
  rt60: "s",
  truncation: "words",
  tempo: "words/min",
  overlap: "s",
  speakers: "speakers",
  f0: "Hz",
};

function MetricCard({ name, value, unit }) {
  const measured = value !== null && value !== undefined;
  return (
    <li className="metric-card">
      <strong>{name}</strong>: {measured ? `${value} ${unit ?? ""}` : "not measured"}
    </li>
  );
}

function toY(amplitude) {
  return 50 - amplitude * 50;
}

// Map a value in [lo, hi] onto 0..size. Pass size first and a flipped range to
// invert an axis (SVG's y grows downward).
function scale(value, lo, hi, size) {
  return ((value - lo) / (hi - lo || 1)) * size;
}

function SpectrumChart({ band }) {
  const freqs = band.spectrum_freqs_hz;
  const db = band.spectrum_db;
  const maxFreq = freqs[freqs.length - 1];
  const lo = Math.min(...db);
  const hi = Math.max(...db);
  const points = freqs
    .map((f, i) => `${scale(f, 0, maxFreq, 800)},${100 - scale(db[i], lo, hi, 100)}`)
    .join(" ");
  const cutoffX = scale(band.effective_cutoff_hz, 0, maxFreq, 800);
  return (
    <div>
      <p>
        Spectrum: where the energy stops (cutoff {band.effective_cutoff_khz} kHz,{" "}
        {band.cliff_found ? `drop ${band.cliff_drop_db} dB` : "no cliff"})
      </p>
      <svg width="800" height="100">
        <polyline points={points} fill="none" stroke="blue" />
        <line x1={cutoffX} x2={cutoffX} y1="0" y2="100" stroke="red" strokeDasharray="4" />
      </svg>
    </div>
  );
}

const PITCH_MIN_HZ = 50;
const PITCH_MAX_HZ = 500;

function PitchChart({ pitch, duration }) {
  // A null in the track is an unvoiced frame: lift the pen and start a new
  // segment so the line breaks instead of joining across the gap.
  let path = "";
  let penDown = false;
  pitch.track_hz.forEach((hz, i) => {
    if (hz === null) {
      penDown = false;
      return;
    }
    const x = scale(pitch.track_times_s[i], 0, duration, 800);
    const y = 100 - scale(hz, PITCH_MIN_HZ, PITCH_MAX_HZ, 100);
    path += `${penDown ? "L" : "M"}${x},${y} `;
    penDown = true;
  });
  const medianY = 100 - scale(pitch.f0_median_hz, PITCH_MIN_HZ, PITCH_MAX_HZ, 100);
  return (
    <div>
      <p>Pitch over speech (median {pitch.f0_median_hz} Hz)</p>
      <svg width="800" height="100">
        <path d={path} fill="none" stroke="purple" />
        <line x1="0" x2="800" y1={medianY} y2={medianY} stroke="gray" strokeDasharray="4" />
      </svg>
    </div>
  );
}

function AnalysisCheckbox({ analysis, onToggle }) {
  return (
    <li>
      <label>
        <input type="checkbox" onChange={() => onToggle(analysis.name)} />{" "}
        {analysis.label}
      </label>
    </li>
  );
}

function App() {
  const [analyses, setAnalyses] = useState([])
  const [selected, setSelected] = useState([])
  const [file, setFile] = useState(null)
  const [digest, setDigest] = useState(null)
  const [error, setError] = useState(null)
  const [runId, setRunId] = useState(null)
  const [run, setRun] = useState(null)
  const [wave, setWave] = useState(null)
  useEffect(() => {
    fetch("http://localhost:8000/v1/analyses")
      .then((response) => response.json())
      .then((data) => setAnalyses(data.analyses));
  }, []);
  useEffect(() => {
    if (!runId) return;
    const timer = setInterval(() => {
      fetch(`http://localhost:8000/v1/runs/${runId}`)
        .then((response) => response.json())
        .then((data) => {
          setRun(data);
          if (data.status === "done" || data.status === "failed") {
            clearInterval(timer)
          }
        });
    }, 1000);
    return () => clearInterval(timer);
  }, [runId]);
  useEffect(() => {
    if (!digest) return;
    fetch(`http://localhost:8000/v1/files/${digest}/waveform`)
      .then((response) => response.json())
      .then((data) => setWave(data))
  }, [digest]);
  function toggle(name) {
    if (selected.includes(name)) {
      setSelected(selected.filter((n) => n !== name));
    } else {
      setSelected([...selected, name]);
    }
  }
  function upload() {
    const form = new FormData();
    form.append("file", file);
    fetch("http://localhost:8000/v1/files", { method: "POST", body: form })
      .then((response) => response.json()).then((data) => setDigest(data.digest))
  }
  async function startRun() {
    const response = await
      fetch(`http://localhost:8000/v1/files/${digest}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ analyses: selected }),
      })
    const data = await response.json();
    if (!response.ok) {
      if (typeof data.detail === "string") {
        setError(data.detail);
      } else {
        setError(data.detail[0].msg);
      }
    } else {
      setError(null);
      setRunId(data.run_id)
    }
  }

  // Seconds to pixels: the drawing is 800 wide and the file lasts duration_s.
  const pxPerSecond = wave ? 800 / wave.duration_s : 0;
  const results = (run && run.results) || {};
  const speechSpans = (results.vad && results.vad.speech_spans) || [];
  const clipPositions = (results.clipping && results.clipping.run_positions_s) || [];

  return (
    <div>
      <h1>Audio dashboard: {selected.length} selected</h1>
      <input type="file" onChange={(event) => setFile(event.target.files[0])} />
      <p>{file && file.name}</p>
      <button onClick={upload}>Upload</button>
      <p>{digest && digest}</p>
      <svg width="800" height="100">
        {wave && speechSpans.map(([start, end]) => (
          <rect
            key={start}
            x={start * pxPerSecond}
            y="0"
            width={(end - start) * pxPerSecond}
            height="100"
            fill="green"
            opacity="0.2"
          />
        ))}
        {wave && wave.max.map((maxValue, index) => (
          <line
            key={index}
            x1={index * (800 / wave.points)}
            x2={index * (800 / wave.points)}
            y1={toY(maxValue)}
            y2={toY(wave.min[index])}
            stroke="blue"
          />))}
        {wave && clipPositions.map((t) => (
          <line
            key={t}
            x1={t * pxPerSecond}
            x2={t * pxPerSecond}
            y1="0"
            y2="100"
            stroke="red"
            opacity="0.6"
          />
        ))}
      </svg>
      
      <button onClick={startRun}>Run</button>
      <p>{error}</p>
      <p>{runId}</p>
      {run && (
        <div>
          <progress value={run.completed} max={run.total} />
          <p>
            {run.completed} / {run.total} {run.status}
          </p>
          {run.status === "failed" && <p>{run.error}</p>}
          {run.status === "done" && (
            <ul>
              {Object.entries(run.results).map(([name, result]) => (
                <MetricCard key={name} name={name} value={result.usable === false ? null : result[MAIN_KEY[name]]} unit={UNIT[name]} />
              ))}
            </ul>
          )}
        </div>
      )}
      <ul>
        {analyses.map((a) => (
          <AnalysisCheckbox key={a.name} analysis={a} onToggle={toggle} />
        ))}
      </ul>
    </div >
  );
}

export default App
