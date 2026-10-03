import type { CapabilityStatus } from './api';

export type Workflow = 'local-training' | 'online-training' | 'local-playtest' | 'matchmaking';

export function workflowBlockers(status: CapabilityStatus | null, workflow: Workflow): string[] {
  if (!status) return ['Checking local capabilities…'];
  if (workflow === 'local-training') {
    return status.torch.ready ? [] : ['Install the deep-learning extra to enable PyTorch.'];
  }
  if (workflow === 'online-training') {
    const blockers: string[] = [];
    if (!status.hosted.api_key_configured) blockers.push('Add DEEPDECK_API_KEY to your .env.');
    if (!status.hosted.trajectory_training) blockers.push(status.hosted.reason);
    return blockers;
  }
  if (workflow === 'matchmaking') {
    const blockers: string[] = [];
    if (!status.sdk.ready) blockers.push('Install DeepDeckAgent SDK.');
    if (!status.hosted.api_key_configured) blockers.push('Create your account API key and add it to the project .env.');
    return blockers;
  }
  const blockers: string[] = [];
  if (!status.sdk.ready) blockers.push('Install DeepDeckAgent SDK.');
  if (!status.engine.source_available) blockers.push('Synchronize DeepDeckEngine.');
  else if (!status.engine.healthy) blockers.push(`Start DeepDeckEngine at ${status.engine.url}.`);
  if (!status.pixi.source_available) blockers.push('Synchronize DeepDeckPixi.');
  else if (!status.pixi.built) blockers.push('Prepare DeepDeckPixi from the local runtime panel.');
  return blockers;
}
