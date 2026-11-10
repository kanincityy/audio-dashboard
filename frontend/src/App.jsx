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

  return (
    <div>
      <h1>Audio dashboard: {selected.length} selected</h1>
      <input type="file" onChange={(event) => setFile(event.target.files[0])} />
      <p>{file && file.name}</p>
      <button onClick={upload}>Upload</button>
      <p>{digest && digest}</p>
      <svg width="800" height="100">
        {wave && wave.max.map((maxValue, index) => (
          <line
            key={index}
            x1={index * (800 / wave.points)}
            x2={index * (800 / wave.points)}
            y1={toY(maxValue)}
            y2={toY(wave.min[index])}
            stroke="blue"
          />))}
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
