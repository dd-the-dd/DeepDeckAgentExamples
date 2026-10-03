import { describe, expect, it } from 'vitest';

import type { CapabilityStatus } from './api';
import { workflowBlockers } from './readiness';

const ready = {
  controller: { ready: true, version: '0.2.0' },
  paths: { project: '/project', trajectory: '/project/.deepdeck/trajectories/decisions.jsonl', checkpoints: '/project/.deepdeck/checkpoints' },
  sdk: { ready: true },
  torch: { ready: true },
  engine: { source_available: true, revision: 'abc', pinned_revision: 'abc', synced: true, dirty: false, built: true, url: 'http://127.0.0.1:8787', healthy: true },
  pixi: { source_available: true, built: true, build_present: true, built_revision: 'def', revision: 'def', pinned_revision: 'def', synced: true, dirty: false },
  hosted: { api_key_configured: false, trajectory_training: false, reason: 'Not published.' },
  workflows: {},
} satisfies CapabilityStatus;

describe('workflowBlockers', () => {
  it('allows local training when torch is ready', () => {
    expect(workflowBlockers(ready, 'local-training')).toEqual([]);
  });

  it('does not confuse hosted inference with training', () => {
    expect(workflowBlockers(ready, 'online-training')).toEqual([
      'Add DEEPDECK_API_KEY to your .env.',
      'Not published.',
    ]);
  });

  it('guides matchmaking to the account key without requiring hosted training', () => {
    expect(workflowBlockers(ready, 'matchmaking')).toEqual([
      'Create your account API key and add it to the project .env.',
    ]);
  });

  it('allows local playtest with healthy locally modified runtimes', () => {
    const localRuntime = {
      ...ready,
      engine: { ...ready.engine, revision: 'newer-local', synced: false, dirty: true },
      pixi: { ...ready.pixi, revision: 'newer-local', synced: false, dirty: true },
    };

    expect(workflowBlockers(localRuntime, 'local-playtest')).toEqual([]);
  });
});
