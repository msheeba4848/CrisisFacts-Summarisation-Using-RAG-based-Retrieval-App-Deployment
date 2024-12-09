import React, { useState } from "react";
import "./styles.css";

function App() {
  const [query, setQuery] = useState("");

  const handleSummarize = () => {
    alert(`Your query is: "${query}"`);
  };

  return (
    <div className="container">
      <h1>Summarization System</h1>
      <h2>Welcome to the Summarization System</h2>
      <input
        type="text"
        placeholder="Enter your query here..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button onClick={handleSummarize}>Summarize</button>
    </div>
  );
}

export default App;

