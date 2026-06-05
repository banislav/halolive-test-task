import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

const CONFIG_KEY = 'deep_agents_web_config';
const THEME_KEY = 'deep_agents_web_theme';

export const DEFAULT_CONFIG = {
  apiBase: 'http://localhost:8000',
  mockMode: true,
};

const AppConfigContext = createContext(null);

export function AppConfigProvider({ children }) {
  const [config, setConfigState] = useState(() => readJson(CONFIG_KEY, DEFAULT_CONFIG));
  const [theme, setTheme] = useState(() => localStorage.getItem(THEME_KEY) || 'light');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const setConfig = useCallback((nextConfig) => {
    setConfigState(nextConfig);
    localStorage.setItem(CONFIG_KEY, JSON.stringify(nextConfig));
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((current) => (current === 'light' ? 'dark' : 'light'));
  }, []);

  const value = useMemo(
    () => ({ config, setConfig, theme, toggleTheme }),
    [config, setConfig, theme, toggleTheme],
  );

  return <AppConfigContext.Provider value={value}>{children}</AppConfigContext.Provider>;
}

export function useAppConfig() {
  const value = useContext(AppConfigContext);
  if (value == null) {
    throw new Error('useAppConfig must be used inside AppConfigProvider');
  }
  return value;
}

function readJson(key, fallback) {
  try {
    return { ...fallback, ...(JSON.parse(localStorage.getItem(key)) || {}) };
  } catch {
    return fallback;
  }
}
