import { Settings, X } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';

import { useAppConfig } from '../lib/config.jsx';

export function SettingsPanel({ open, onClose }) {
  const { config, setConfig } = useAppConfig();
  const [local, setLocal] = useState(config);

  useEffect(() => {
    if (open) setLocal(config);
  }, [config, open]);

  const update = useCallback((key, value) => {
    setLocal((current) => ({ ...current, [key]: value }));
  }, []);

  const save = useCallback(() => {
    setConfig(local);
    onClose();
  }, [local, onClose, setConfig]);

  return (
    <>
      <div className={`overlay${open ? ' open' : ''}`} onClick={onClose} />
      <aside className={`settings-panel${open ? ' open' : ''}`}>
        <div className="settings-head">
          <div className="inline">
            <Settings size={16} />
            <span className="settings-title">Settings</span>
          </div>
          <button className="btn-icon icon-only" onClick={onClose}>
            <X size={15} />
          </button>
        </div>
        <div className="settings-body">
          <div className="mock-row">
            <div>
              <div className="mock-label">Mock mode</div>
              <div className="mock-desc">Use local sample data instead of the Python API</div>
            </div>
            <button
              className={`switch-btn${local.mockMode ? ' on' : ''}`}
              onClick={() => update('mockMode', !local.mockMode)}
              aria-label="Toggle mock mode"
            />
          </div>
          <div className="field-sep" />
          <div className="field" style={{ opacity: local.mockMode ? 0.45 : 1 }}>
            <label>API Base URL</label>
            <input
              type="text"
              value={local.apiBase}
              onChange={(event) => update('apiBase', event.target.value)}
              placeholder="http://localhost:8000"
              spellCheck="false"
              disabled={local.mockMode}
            />
            <span className="field-hint">Python FastAPI bridge origin</span>
          </div>
          <button className="btn btn-primary" onClick={save}>
            Save settings
          </button>
        </div>
      </aside>
    </>
  );
}
