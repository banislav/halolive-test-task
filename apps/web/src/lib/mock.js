const MOCK_EVENTS = [
  {
    from: 'runtime_session',
    to: 'user',
    type: 'progress',
    correlation_id: 'mock-session',
    payload: { status: 'session_started', execution_plan_id: 'EP-mock' },
  },
  {
    from: 'dispatcher',
    to: 'worker',
    type: 'request',
    correlation_id: 'T1',
    payload: { task_id: 'T1' },
  },
  {
    from: 'worker',
    to: 'plan_tracker',
    type: 'result',
    correlation_id: 'T1',
    payload: { result: { task_id: 'T1', output: { summary: 'Drafted runtime summary.' } } },
  },
  {
    from: 'judge',
    to: 'plan_tracker',
    type: 'verdict',
    correlation_id: 'T1',
    payload: { verdict: { task_id: 'T1', verdict: 'pass', recommendation: 'advance' } },
  },
];

export function createMockSession(prompt) {
  const sessionId = `mock-${Date.now()}`;
  const snapshot = {
    session_id: sessionId,
    status: 'completed',
    execution_plan_id: 'EP-mock',
    current_task_id: null,
    plan_state: {
      objective: { raw: prompt },
      status: 'completed',
      task_statuses: { T1: 'completed', T2: 'completed' },
    },
    results: {
      T1: {
        task_id: 'T1',
        output: {
          section: 'Discovery and execution flow',
          content:
            'The runtime converts a raw prompt into a discovery plan, turns that into an execution plan, and dispatches work through an async session.',
        },
        artifacts: [],
      },
      T2: {
        task_id: 'T2',
        output: {
          section: 'Interactive session',
          content:
            'The session streams runtime messages and accepts read-only prompts while background task execution continues at safe boundaries.',
        },
        artifacts: [],
      },
    },
    runtime_commands: [],
    command_results: [],
    prompt_results: [],
    pending_prompt_ids: [],
    memory_record_count: 8,
  };
  localStorage.setItem(
    `deep_agents_mock_session:${sessionId}`,
    JSON.stringify({ snapshot, events: MOCK_EVENTS }),
  );
  return { sessionId, snapshot };
}

export function readMockSession(sessionId) {
  try {
    return JSON.parse(localStorage.getItem(`deep_agents_mock_session:${sessionId}`));
  } catch {
    return null;
  }
}

export function appendMockPrompt(sessionId, content) {
  const record = readMockSession(sessionId);
  if (record == null) {
    return null;
  }
  const promptId = `prompt-${Date.now()}`;
  const result = {
    prompt: {
      id: promptId,
      content,
      priority: 3,
      category: 'content_reasoning',
      metadata: {},
      queued_at: new Date().toISOString(),
    },
    classification: {
      prompt_id: promptId,
      category: 'content_reasoning',
      priority: 3,
      reasoning: 'Mock read-only progress question.',
    },
    response: {
      prompt_id: promptId,
      answer: 'The mock session has completed two tasks and recorded runtime progress.',
      referenced_task_ids: ['T1', 'T2'],
      referenced_artifact_ids: [],
    },
    commands: [],
  };
  record.snapshot.prompt_results = [...record.snapshot.prompt_results, result];
  record.events = [
    ...record.events,
    {
      from: 'prompt_queue',
      to: 'content_reasoning_agent',
      type: 'prompt',
      correlation_id: promptId,
      payload: { prompt_result: result },
    },
  ];
  localStorage.setItem(`deep_agents_mock_session:${sessionId}`, JSON.stringify(record));
  return { prompt: result.prompt, snapshot: record.snapshot };
}
