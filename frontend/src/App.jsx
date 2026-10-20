import { useState, useEffect } from 'react'
import './App.css'

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
        {analyses.map((a) => <li key={a.name}><label><input type="checkbox" onChange={() => toggle(a.name)} /> {a.label}</label></li>)}
      </ul>
    </div>
  );
}

export default App
