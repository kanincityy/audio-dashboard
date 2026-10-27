import { useState, useEffect } from 'react'
import './App.css'

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
  useEffect(() => {
    fetch("http://localhost:8000/v1/analyses")
      .then((response) => response.json())
      .then((data) => setAnalyses(data.analyses));
  }, []);
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
    }
  }

  return (
    <div>
      <h1>Audio dashboard: {selected.length} selected</h1>
      <input type="file" onChange={(event) => setFile(event.target.files[0])} />
      <p>{file && file.name}</p>
      <button onClick={upload}>Upload</button>
      <p>{digest && digest}</p>
      <button onClick={startRun}>Run</button>
      <p>{error}</p>
      <ul>
        {analyses.map((a) => (
          <AnalysisCheckbox key={a.name} analysis={a} onToggle={toggle} />
        ))}
      </ul>
    </div>
  );
}

export default App
