import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  Outlet,
  RouterProvider,
  createRootRoute,
  createRoute,
  createRouter,
} from '@tanstack/react-router';
import React from 'react';
import { createRoot } from 'react-dom/client';

import { AppShell } from './components/app-shell.jsx';
import { PromptRunnerPage } from './routes/prompt-runner.jsx';
import { SessionPage } from './routes/session.jsx';
import './styles.css';

const queryClient = new QueryClient();

const rootRoute = createRootRoute({
  component: () => (
    <AppShell>
      <Outlet />
    </AppShell>
  ),
});

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: PromptRunnerPage,
});

const sessionRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/sessions/$sessionId',
  component: SessionPage,
});

const router = createRouter({
  routeTree: rootRoute.addChildren([indexRoute, sessionRoute]),
});

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </React.StrictMode>,
);
