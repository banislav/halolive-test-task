import { useMutation, useQuery } from '@tanstack/react-query';
import { useParams } from '@tanstack/react-router';
import { LoaderCircle, Send } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';

import { SessionOutput } from '../components/session-output.jsx';
import { eventsUrl, fetchSnapshot, submitPrompt } from '../lib/api.js';
import { useAppConfig } from '../lib/config.jsx';
import { appendMockPrompt, readMockSession } from '../lib/mock.js';

export function SessionPage() {
  const { sessionId } = useParams({ strict: false });
  const { config } = useAppConfig();
  const [events, setEvents] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [error, setError] = useState(null);
  const isMock = sessionId.startsWith('mock-') || config.mockMode;

  const snapshotQuery = useQuery({
    queryKey: ['session-snapshot', sessionId, isMock],
    queryFn: () => {
      if (isMock) return readMockSession(sessionId)?.snapshot || null;
      return fetchSnapshot(config, sessionId);
    },
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'running' || status === 'paused' ? 1500 : false;
    },
  });

  useEffect(() => {
    setEvents([]);
    if (isMock) {
      const record = readMockSession(sessionId);
      setEvents(record?.events || []);
      return undefined;
    }
    const source = new EventSource(eventsUrl(config, sessionId));
    source.onmessage = (event) => {
      setEvents((current) => [...current, JSON.parse(event.data)]);
      snapshotQuery.refetch();
    };
    source.onerror = () => {
      source.close();
    };
    return () => source.close();
  }, [config, isMock, sessionId]);

  const promptMutation = useMutation({
    mutationFn: async (content) => {
      if (isMock) return appendMockPrompt(sessionId, content);
      return submitPrompt(config, sessionId, {
        content,
        priority: 3,
        category: 'content_reasoning',
        metadata: {},
      });
    },
    onSuccess: () => {
      setPrompt('');
      snapshotQuery.refetch();
      if (isMock) {
        const record = readMockSession(sessionId);
        setEvents(record?.events || []);
      }
    },
    onError: (err) => setError(err.message || 'Prompt failed'),
  });

  const submit = useCallback(() => {
    if (!prompt.trim()) return;
    setError(null);
    promptMutation.mutate(prompt.trim());
  }, [prompt, promptMutation]);

  const snapshot = snapshotQuery.data;

  return (
    <>
      <div className="card prompt-card">
        <div className="prompt-label">Session Prompt</div>
        <textarea
          className="prompt-textarea"
          placeholder="Ask for read-only progress, results, or artifacts..."
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={(event) => {
            if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
              event.preventDefault();
              submit();
            }
          }}
        />
        <div className="prompt-footer">
          <span className="char-count">{sessionId}</span>
          <button className="btn btn-primary" onClick={submit} disabled={!prompt.trim() || promptMutation.isPending}>
            {promptMutation.isPending ? <LoaderCircle size={14} className="spin" /> : <Send size={14} />}
            Ask
          </button>
        </div>
      </div>
      {(error || snapshotQuery.error) && (
        <div className="card error-card">
          <p className="error-text">{error || snapshotQuery.error.message}</p>
        </div>
      )}
      <SessionOutput snapshot={snapshot} events={events} />
    </>
  );
}
