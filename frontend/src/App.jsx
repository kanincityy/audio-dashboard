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
    <h1>Audio dashboard: {analyses.length} analyses</h1>)
}

export default App
