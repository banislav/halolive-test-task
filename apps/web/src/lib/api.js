export async function planFromPrompt(config, payload) {
  return request(config, '/api/plans/from-prompt', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function createSession(config, payload) {
  return request(config, '/api/sessions', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function fetchSnapshot(config, sessionId) {
  return request(config, `/api/sessions/${encodeURIComponent(sessionId)}/snapshot`);
}

export async function submitPrompt(config, sessionId, payload) {
  return request(config, `/api/sessions/${encodeURIComponent(sessionId)}/prompts`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function eventsUrl(config, sessionId) {
  return `${trimBase(config.apiBase)}/api/sessions/${encodeURIComponent(sessionId)}/events`;
}

async function request(config, path, options = {}) {
  const response = await fetch(`${trimBase(config.apiBase)}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      // The status text is enough when the server did not return JSON.
    }
    throw new Error(`HTTP ${response.status}: ${detail}`);
  }
  return response.json();
}

function trimBase(value) {
  return (value || '').replace(/\/$/, '');
}
