import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [analyses, setAnalyses] = useState([])
  useEffect(() => {
    fetch("http://localhost:8000/v1/analyses")
      .then((response) => response.json())
      .then((data) => setAnalyses(data.analyses));
  }, []);

  return (
    <div>
      <h1>Audio dashboard</h1>
      <ul>
        {analyses.map((a) => <li key={a.name}>{a.label}</li>)}
      </ul>
    </div>
  );
}

export default App
