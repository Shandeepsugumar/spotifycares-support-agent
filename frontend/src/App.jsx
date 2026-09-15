import { useState } from 'react'
import './App.css'

function App() {
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!message.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'

    try {
      const response = await fetch(`${apiUrl}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'An error occurred connecting to the backend.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header>
        <h1>SpotifyCares Support Agent Demo</h1>
        <p>Interactive testing for the RAG-powered customer support agent.</p>
      </header>

      <main>
        <form onSubmit={handleSubmit} className="input-section">
          <label htmlFor="customer-message">Customer Message:</label>
          <textarea
            id="customer-message"
            rows="4"
            placeholder="e.g., My Spotify app keeps crashing when I play downloaded songs!"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            disabled={loading}
          />
          <button type="submit" disabled={loading || !message.trim()}>
            {loading ? 'Analyzing...' : 'Analyze Message'}
          </button>
        </form>

        {error && <div className="error-banner"><strong>Error:</strong> {error}</div>}

        {result && (
          <div className="result-panel">
            <h2>Agent Decision</h2>
            
            <div className="result-field">
              <span className="label">Action:</span>
              <span className={`badge ${result.action === 'escalate' ? 'escalate' : 'auto'}`}>
                {result.action === 'escalate' ? 'Escalate to Human' : 'Auto-Handle'}
              </span>
            </div>

            <div className="result-field">
              <span className="label">Predicted Intent:</span>
              <span className="value">{result.intent}</span>
            </div>

            <div className="result-field">
              <span className="label">Draft Reply:</span>
              <div className="draft-reply">{result.draft_reply}</div>
            </div>

            {result.reason && (
              <div className="result-field">
                <span className="label">Reasoning:</span>
                <span className="value">{result.reason}</span>
              </div>
            )}

            <div className="result-field">
              <span className="label">Evidence Cites:</span>
              <div className="evidence-list">
                {result.evidence_ids && result.evidence_ids.length > 0 ? (
                  result.evidence_ids.map((id, i) => <span key={i} className="evidence-id">{id}</span>)
                ) : (
                  <span className="value">None</span>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
