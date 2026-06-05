import { Code, Copy, FileText } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';

import { jsonHtml, markdownToHtml } from '../lib/format.js';
import { TaskCard } from './task-card.jsx';

export function SessionOutput({ snapshot, events }) {
  const [viewMode, setViewMode] = useState('readable');
  const tasks = useMemo(() => Object.values(snapshot?.results || {}), [snapshot]);
  const promptResponses = (snapshot?.prompt_results || []).filter((item) => item.response);
  const latestPromptResponse = promptResponses[promptResponses.length - 1]?.response;

  const copyAll = useCallback(() => {
    navigator.clipboard
      .writeText(JSON.stringify({ snapshot, events }, null, 2))
      .catch(() => {});
  }, [events, snapshot]);

  if (!snapshot) return null;

  return (
    <section className="output-section">
      <div className="section-divider">
        <span className="section-divider-label">Runtime</span>
      </div>
      <div className="card summary-card">
        <div className="summary-header">
          <span className="summary-id">{snapshot.session_id}</span>
          <span className={`status-pill ${snapshot.status}`}>{snapshot.status}</span>
        </div>
        <p className="summary-answer">
          {latestPromptResponse?.answer ||
            `Execution plan ${snapshot.execution_plan_id || 'pending'} is ${snapshot.status}.`}
        </p>
        <div className="ref-ids">
          <span className="ref-label">Memory</span>
          <span className="chip">{snapshot.memory_record_count}</span>
          <span className="ref-label">Events</span>
          <span className="chip">{events.length}</span>
        </div>
      </div>

      <div className="tasks-header">
        <span className="tasks-title">Task Outputs ({tasks.length})</span>
        <div className="tasks-actions">
          <div className="seg">
            <button
              className={`seg-btn${viewMode === 'readable' ? ' active' : ''}`}
              onClick={() => setViewMode('readable')}
            >
              <FileText size={12} /> Readable
            </button>
            <button
              className={`seg-btn${viewMode === 'json' ? ' active' : ''}`}
              onClick={() => setViewMode('json')}
            >
              <Code size={12} /> JSON
            </button>
          </div>
          <button className="btn-icon" onClick={copyAll}>
            <Copy size={14} /> Copy all
          </button>
        </div>
      </div>
      <div className="task-grid">
        {tasks.map((task, index) => (
          <TaskCard key={task.task_id} task={task} viewMode={viewMode} animDelay={index * 60} />
        ))}
        {tasks.length === 0 && <EmptyCard text="Waiting for task results." />}
      </div>

      <div className="tasks-header events-header">
        <span className="tasks-title">Event Stream ({events.length})</span>
      </div>
      <div className="event-list">
        {events.map((event, index) => (
          <div key={`${event.correlation_id}-${index}`} className="event-row">
            <span className="event-type">{event.type}</span>
            <span className="event-route">
              {event.from}
              {' -> '}
              {event.to}
            </span>
            <span className="event-correlation">{event.correlation_id}</span>
          </div>
        ))}
      </div>

      {viewMode === 'json' && (
        <div className="card json-panel">
          <div className="json-view" dangerouslySetInnerHTML={{ __html: jsonHtml(snapshot) }} />
        </div>
      )}
    </section>
  );
}

function EmptyCard({ text }) {
  return (
    <div className="card task-card">
      <div className="readable" dangerouslySetInnerHTML={{ __html: markdownToHtml(text) }} />
    </div>
  );
}
