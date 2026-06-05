import { Link } from '@tanstack/react-router';
import { Moon, Settings, Sun } from 'lucide-react';
import { useState } from 'react';

import { AppConfigProvider, useAppConfig } from '../lib/config.jsx';
import { SettingsPanel } from './settings-panel.jsx';

export function AppShell({ children }) {
  return (
    <AppConfigProvider>
      <ShellInner>{children}</ShellInner>
    </AppConfigProvider>
  );
}

function ShellInner({ children }) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { config, theme, toggleTheme } = useAppConfig();

  return (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="topbar-brand">
          <span className="topbar-dot" />
          <span className="topbar-title">Deep Agents Runtime</span>
          {config.mockMode && <span className="badge-mock">MOCK</span>}
        </Link>
        <div className="topbar-actions">
          <button className="btn-icon icon-only" onClick={toggleTheme} title="Toggle theme">
            {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
          </button>
          <button
            className={`btn-icon icon-only${settingsOpen ? ' is-copied' : ''}`}
            onClick={() => setSettingsOpen(true)}
            title="Settings"
          >
            <Settings size={16} />
          </button>
        </div>
      </header>
      <main className="main">{children}</main>
      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
