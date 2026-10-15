'use strict';

const percent = (n, total) => (100 * n / total).toFixed(2);
const $ = (selector) => document.querySelector(selector);

function setSelected(attribute, value) {
  document.querySelectorAll(`[${attribute}]`).forEach((button) => {
    const selected = button.getAttribute(attribute) === value;
    button.classList.toggle('active', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
}

function renderReplication(runs) {
  $('#replication-bars').innerHTML = runs.map((run) => {
    const count = (arm) => run.counts[arm].test_multi;
    return `<div class="comparison-group"><div class="seed-label">Seed ${run.seed}</div>${['control', 'treatment'].map((arm) => {
      const row = count(arm);
      return `<div class="bar-row"><span>${arm === 'control' ? 'Old' : 'Multi-entity'}</span><div class="bar-track"><i class="${arm}" style="width:${percent(row.complete_families, row.families)}%"></i></div><span>${row.complete_families}/${row.families}</span></div>`;
    }).join('')}<div class="gain-note">+${(100 * run.primary.estimate).toFixed(2)} percentage points</div></div>`;
  }).join('');
}

function renderIntervals(runs) {
  const x = (value) => 135 + (value + 3) * 40;
  const rows = runs.map((run, index) => {
    const y = 40 + index * 65;
    const mean = 100 * run.primary.estimate;
    const [low, high] = run.primary.interval.map((v) => 100 * v);
    return `<text x="0" y="${y + 4}" fill="currentColor">Seed ${run.seed}</text><line x1="${x(low)}" y1="${y}" x2="${x(high)}" y2="${y}" stroke="currentColor" stroke-width="2"/><path d="M${x(low)} ${y - 5}v10M${x(high)} ${y - 5}v10" stroke="currentColor"/><circle cx="${x(mean)}" cy="${y}" r="4" fill="currentColor"/><text x="${x(mean)}" y="${y - 15}" text-anchor="middle" fill="currentColor">${mean.toFixed(2)} [${low.toFixed(2)}, ${high.toFixed(2)}]</text>`;
  }).join('');
  const ticks = [-2, 0, 2, 4, 6].map((n) => `<text x="${x(n)}" y="140" text-anchor="middle" fill="currentColor">${n}</text>`).join('');
  $('#loss-intervals').innerHTML = `<svg viewBox="0 0 560 165" role="img" aria-label="Seed 1601 gain 0.79 points, interval minus 1.85 to 3.70. Seed 1801 gain 3.70, interval 1.06 to 6.35."><line x1="${x(0)}" y1="8" x2="${x(0)}" y2="123" stroke="#555" stroke-dasharray="3 4"/>${rows}${ticks}<text x="${x(2)}" y="161" text-anchor="middle" fill="currentColor" font-size="10">Difference in percentage points · zero = no observed gain</text></svg>`;
}

function renderReliability(runs, rep) {
  const run = runs.find((r) => r.repetition === rep);
  const counts = run.counts.treatment;
  const definitions = [
    ['Individual base answers', 'test_multi', 'correct_questions', 'questions', 'questions'],
    ['Every person in a base world', 'test_multi', 'complete_families', 'families', 'worlds'],
    ['Every person before + after', 'test_paired', 'complete_families', 'families', 'paired worlds'],
    ['Every question, all nine variants', 'test_multi_transformations', 'complete_families', 'families', 'families'],
  ];
  $('#reliability-bars').innerHTML = definitions.map(([label, split, numerator, denominator, unit]) => {
    const row = counts[split];
    const value = percent(row[numerator], row[denominator]);
    return `<div class="threshold-row"><div><b>${label}</b><small>${row[numerator]}/${row[denominator]} ${unit}</small></div><div class="bar-track"><i style="width:${value}%"></i></div><strong>${value}%</strong></div>`;
  }).join('');
  setSelected('data-reliability', rep);
}

function renderCase(examples, rep) {
  const example = examples[rep];
  const removed = $('#remove-premise').checked;
  const variant = removed ? 'support_removal' : 'base';
  $('#case-question').textContent = example.question;
  const gold = example.outputs.parent[variant].gold;
  $('#case-gold').textContent = gold;
  $('#case-gold').className = `class-label ${gold.toLowerCase()}`;
  const person = example.question.split(' ')[0];
  const relevant = example.base_facts.filter((fact) => fact.startsWith(`${person} `));
  const focus = $('#case-focus');
  focus.replaceChildren();
  relevant.forEach((fact) => {
    const span = document.createElement('span');
    span.className = `fact${removed && example.removed_facts.includes(fact) ? ' removed' : ''}`;
    span.textContent = fact;
    focus.append(span);
  });
  if (removed) example.added_facts.forEach((fact) => {
    const span = document.createElement('span');
    span.className = 'fact added';
    span.textContent = `+ ${fact}`;
    focus.append(span);
  });
  const rules = $('#case-rules');
  rules.replaceChildren();
  example.rules.forEach((rule) => {
    const p = document.createElement('p');
    p.textContent = rule;
    rules.append(p);
  });
  const names = {parent: 'Before extra training', control: 'Answer loss only', treatment: 'Answer + consistency loss'};
  $('#case-predictions').innerHTML = Object.entries(names).map(([arm, label]) => {
    const output = example.outputs[arm][variant];
    const correct = output.prediction === output.gold;
    return `<tr><td>${label}</td><td><span class="class-label ${output.prediction.toLowerCase()}">${output.prediction}</span></td><td>${(100 * output.confidence).toFixed(2)}%</td><td class="verdict-${correct ? 'correct' : 'wrong'}">${correct ? 'YES' : 'NO'}</td></tr>`;
  }).join('');
  $('#case-interpretation').textContent = removed
    ? `the correct answer is UNKNOWN. all three models still answer TRUE. the consistency model is wrong with ${(100 * example.outputs.treatment.support_removal.confidence).toFixed(2)}% confidence.`
    : 'all three models correctly answer TRUE. both required facts are present. remove verification to see whether the answer follows the evidence.';
  const full = $('#case-full');
  full.replaceChildren();
  const facts = removed ? example.base_facts.filter((fact) => !example.removed_facts.includes(fact)).concat(example.added_facts) : example.base_facts;
  const list = document.createElement('ul');
  facts.forEach((fact) => {
    const li = document.createElement('li');
    li.textContent = fact;
    list.append(li);
  });
  const id = document.createElement('p');
  id.className = 'mono';
  id.textContent = `Frozen family: ${example.family_id}`;
  full.append(list, id);
  setSelected('data-case', rep);
}

async function main() {
  const response = await fetch('assets/results.json');
  if (!response.ok) throw new Error(`Evidence data returned HTTP ${response.status}`);
  const data = await response.json();
  renderReplication(data.studies.v7);
  renderIntervals(data.studies.v8);
  renderReliability(data.studies.v8, 'r1');
  renderCase(data.examples, 'r2');
  document.querySelectorAll('[data-reliability]').forEach((button) => {
    button.addEventListener('click', () => renderReliability(data.studies.v8, button.dataset.reliability));
  });
  $('#remove-premise').addEventListener('change', () => renderCase(data.examples, 'r2'));
  document.documentElement.dataset.ready = 'true';
}

$('#print').addEventListener('click', () => window.print());
const words = document.querySelector('.article-body').textContent.trim().split(/\s+/).length;
$('#reading-time').textContent = `${Math.ceil(words / 220)} min read`;
main().catch((error) => {
  console.error(error);
  const message = document.createElement('p');
  message.className = 'data-error';
  message.textContent = 'The saved figure data could not be loaded. Please reload the page. The article tables and evidence links remain available.';
  $('#replication-bars').append(message);
  document.documentElement.dataset.ready = 'error';
});
