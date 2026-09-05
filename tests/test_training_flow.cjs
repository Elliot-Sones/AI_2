const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const file = path.join(__dirname, '../docs/training-flow/index.html');
function load() {
  assert.ok(fs.existsSync(file), 'The standalone training-flow explainer must exist');
  const html = fs.readFileSync(file, 'utf8');
  const script = html.match(/<script id="flow-model">([\s\S]*?)<\/script>/);
  assert.ok(script, 'The timing model must be independently testable');
  const context = {};
  vm.runInNewContext(script[1], context);
  return { html, model: context.TrainingFlowModel };
}

test('round model partitions the measured environment interval without double counting', () => {
  const { model } = load();
  const stages = model.round;
  const environment = stages.filter(s => ['dispatch', 'world', 'collect'].includes(s.id));
  assert.ok(Math.abs(environment.reduce((sum, s) => sum + s.ms, 0) - 5.502494531356206) < 1e-9);
  assert.ok(Math.abs(model.roundMs - stages.reduce((sum, s) => sum + s.ms, 0)) < 1e-9);
  assert.equal(stages.some(s => s.id === 'update'), false, 'Learning must not be inserted into each round');
});

test('scrubber resolves boundaries and clamps endpoints', () => {
  const { model } = load();
  assert.equal(model.at(model.round, -10).index, 0);
  assert.equal(model.at(model.round, model.round[0].ms).index, 1);
  assert.equal(model.at(model.round, model.roundMs + 10).index, model.round.length - 1);
  for (const stage of model.round) assert.ok(stage.ms > 0);
});

test('episode ending and rollout learning remain distinct and schematic', () => {
  const { model, html } = load();
  const ids = model.story.map(s => s.id);
  assert.ok(ids.indexOf('episode-end') < ids.indexOf('rollout-ready'));
  assert.ok(ids.indexOf('rollout-ready') < ids.indexOf('update'));
  assert.ok(model.story.every(s => s.ms === 1), 'Lifecycle uses event positions, not invented timestamps');
  assert.match(html, /not a recorded episode/i);
  assert.match(html, /bandwidth.*not measured/i);
  assert.match(html, /kernel.*not measured/i);
});

test('payload sizes are shape-derived and do not move game state to VRAM', () => {
  const { model } = load();
  assert.equal(model.bytes.observations, 16 * 2656 * 4);
  assert.equal(model.bytes.actions, 16 * 10 * 4);
  assert.equal(model.bytes.trainingObservations, 8192 * 2656 * 4);
  assert.equal(model.round.find(s => s.id === 'world').link, 'none');
  assert.equal(model.round.find(s => s.id === 'observations').link, 'to-gpu');
  assert.equal(model.round.find(s => s.id === 'actions').link, 'to-cpu');
});

test('CPU detail includes real reward and observation arithmetic, without invented physics timings', () => {
  const { html } = load();
  assert.match(html, /2 × \(previous_distance − current_distance\)/);
  assert.match(html, /max\(-18, min\(18, x\)\)/);
  assert.match(html, /Illustrative numbers, not a recorded trajectory/);
});
