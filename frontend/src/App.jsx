import { useEffect } from 'react'
import './App.css'

function App() {
  useEffect(() => {
    fetch("http://localhost:8000/v1/analyses")
      .then((response) => response.json())
      .then((data) => console.log(data));
  }, []);

  return <h1>Audio dashboard</h1>;
}

export default App
