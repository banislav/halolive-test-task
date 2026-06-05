import { History, LoaderCircle, Send } from 'lucide-react';
import { useCallback, useRef, useState } from 'react';

import { relTime } from '../lib/format.js';

export function PromptCard({ onSubmit, isLoading, history, onLoadHistory }) {
  const [text, setText] = useState('');
  const [historyOpen, setHistoryOpen] = useState(false);
  const textareaRef = useRef(null);

  const submit = useCallback(() => {
    if (!text.trim() || isLoading) return;
    onSubmit(text.trim());
  }, [isLoading, onSubmit, text]);

  return (
    <div>
      <div className="card prompt-card">
        <div className="prompt-label">Prompt</div>
        <textarea
          ref={textareaRef}
          className="prompt-textarea"
          placeholder="Describe a task for the agent runtime..."
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
              event.preventDefault();
              submit();
            }
          }}
          disabled={isLoading}
          autoFocus
        />
        <div className="prompt-footer">
          <span className="char-count">{text.length ? `${text.length} chars` : 'Cmd/Ctrl Enter'}</span>
          <div className="inline">
            {history.length > 0 && (
              <button className="btn btn-ghost compact" onClick={() => setHistoryOpen((open) => !open)}>
                <History size={14} />
                History ({history.length})
              </button>
            )}
            <button className="btn btn-primary" onClick={submit} disabled={!text.trim() || isLoading}>
              {isLoading ? <LoaderCircle size={14} className="spin" /> : <Send size={14} />}
              {isLoading ? 'Running' : 'Run'}
            </button>
          </div>
        </div>
      </div>
      {historyOpen && (
        <div className="history-list">
          {history.map((item) => (
            <button
              key={item.id}
              className="history-item"
              onClick={() => {
                setText(item.prompt);
                setHistoryOpen(false);
                onLoadHistory(item);
                setTimeout(() => textareaRef.current?.focus(), 50);
              }}
            >
              <span className="history-dot" />
              <span className="history-text">{item.prompt}</span>
              <span className="history-meta">{relTime(item.timestamp)}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
