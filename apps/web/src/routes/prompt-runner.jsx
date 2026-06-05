import { useMutation } from '@tanstack/react-query';
import { useNavigate } from '@tanstack/react-router';
import { useCallback, useState } from 'react';

import { PromptCard } from '../components/prompt-card.jsx';
import { useAppConfig } from '../lib/config.jsx';
import { createSession, planFromPrompt } from '../lib/api.js';
import { createMockSession } from '../lib/mock.js';

const HISTORY_KEY = 'deep_agents_prompt_history';

export function PromptRunnerPage() {
  const navigate = useNavigate();
  const { config } = useAppConfig();
  const [history, setHistory] = useState(() => readHistory());
  const [error, setError] = useState(null);

  const runMutation = useMutation({
    mutationFn: async (prompt) => {
      if (config.mockMode) {
        return createMockSession(prompt);
      }
      const plans = await planFromPrompt(config, {
        prompt,
        constraints: [],
        available_tools: [],
        available_skills: [],
        context: {},
      });
      const session = await createSession(config, {
        discovery_plan: plans.discovery_plan,
        execution_plan: plans.execution_plan,
      });
      return { sessionId: session.session_id, snapshot: session.snapshot, plans };
    },
    onSuccess: (result, prompt) => {
      saveHistory(setHistory, { prompt, sessionId: result.sessionId });
      navigate({ to: '/sessions/$sessionId', params: { sessionId: result.sessionId } });
    },
    onError: (err) => setError(err.message || 'Request failed'),
  });

  const submit = useCallback(
    (prompt) => {
      setError(null);
      runMutation.mutate(prompt);
    },
    [runMutation],
  );

  return (
    <>
      <PromptCard
        onSubmit={submit}
        isLoading={runMutation.isPending}
        history={history}
        onLoadHistory={(item) => {
          if (item?.sessionId) {
            navigate({ to: '/sessions/$sessionId', params: { sessionId: item.sessionId } });
          } else {
            setHistory([]);
            localStorage.removeItem(HISTORY_KEY);
          }
        }}
      />
      {error && (
        <div className="card error-card">
          <p className="error-text">{error}</p>
        </div>
      )}
    </>
  );
}

function readHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch {
    return [];
  }
}

function saveHistory(setHistory, item) {
  setHistory((current) => {
    const next = [
      {
        id: String(Date.now()),
        timestamp: new Date().toISOString(),
        ...item,
      },
      ...current,
    ].slice(0, 10);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
    return next;
  });
}
