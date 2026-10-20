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

  return (
    <div>
      <h1>Audio dashboard: {selected.length} selected</h1>
      <ul>
        {analyses.map((a) => (
          <AnalysisCheckbox key={a.name} analysis={a} onToggle={toggle} />
        ))}
      </ul>
    </div>
  );
}

export default App
