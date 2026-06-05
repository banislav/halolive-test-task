import { Code, Copy, FileText } from 'lucide-react';
import { useCallback, useState } from 'react';

import { jsonHtml, markdownToHtml } from '../lib/format.js';

export function TaskCard({ task, viewMode, animDelay = 0 }) {
  const [localView, setLocalView] = useState(null);
  const active = localView || viewMode;
  const output = task.output || {};
  const section = output.section || output.title || task.task_id || task.id;
  const content = output.content || output.summary || output.text || JSON.stringify(output, null, 2);

  const copy = useCallback(() => {
    const text = active === 'json' ? JSON.stringify(task, null, 2) : `${section}\n\n${content}`;
    navigator.clipboard.writeText(text).catch(() => {});
  }, [active, content, section, task]);

  return (
    <article className="card task-card task-card-animate" style={{ animationDelay: `${animDelay}ms` }}>
      <div className="task-card-header">
        <div className="task-card-left">
          <span className="task-id-chip">{task.task_id || task.id}</span>
          <span className="task-section">{section}</span>
        </div>
        <div className="inline">
          <div className="seg">
            <button
              className={`seg-btn${active === 'readable' ? ' active' : ''}`}
              onClick={() => setLocalView('readable')}
              title="Readable"
            >
              <FileText size={12} />
            </button>
            <button
              className={`seg-btn${active === 'json' ? ' active' : ''}`}
              onClick={() => setLocalView('json')}
              title="JSON"
            >
              <Code size={12} />
            </button>
          </div>
          <button className="btn-icon icon-only" onClick={copy} title="Copy task">
            <Copy size={14} />
          </button>
        </div>
      </div>
      <div className="task-content">
        {active === 'json' ? (
          <div className="json-view" dangerouslySetInnerHTML={{ __html: jsonHtml(task) }} />
        ) : (
          <div className="readable" dangerouslySetInnerHTML={{ __html: markdownToHtml(content) }} />
        )}
      </div>
      {(task.artifacts?.length > 0 || task.status) && (
        <div className="chip-row">
          {task.status && <span className="chip green">{task.status}</span>}
          {task.artifacts?.map((artifact, index) => (
            <span key={index} className="chip">
              {typeof artifact === 'string' ? artifact : artifact.id || JSON.stringify(artifact)}
            </span>
          ))}
        </div>
      )}
    </article>
  );
}
