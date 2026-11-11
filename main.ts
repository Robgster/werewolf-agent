// A Werewolf agent for make.fun's werewolf-agent-01.
//
// How it plays: every decision the env waits on goes to claude-opus-5-5 (thinking, effort low) once, with
// the rules, role-specific strategy coaching, a readable digest of that one game's view (players, private
// knowledge, the public log, today's vote tally and who claimed what), and the decision. The reply's JSON is
// then passed through hard rules that the model must not get wrong:
//   - wolves: the second wolf at night copies the partner's pick, so the kill always lands; a wolf never
//     votes to exile its partner (unless the partner is already lost); a wolf never names its partner
//   - seer: never re-inspects a seat it already knows; votes for a werewolf it found while one lives
//   - village: never votes a seat the seer knows is clean
// When the model gives nothing usable (no key pushed, budget spent, an unreadable reply), a scripted player
// answers instead (see scripted*), so no decision is a blind pass.
//
// Deterministic given the model's replies: no clock, and every "random" pick is a hash of MAKEFUN_TRIAL_SEED,
// the game number and the log position. Each request is built from its own game's view alone.

import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';

export const MODEL = 'claude-opus-5-5';
export const EFFORT = 'low';
export const MAX_TOKENS = 2048;
export const ANTHROPIC_VERSION = '2023-06-01';
export const CONCURRENCY = 10;
export const RETRY_WAIT_SEC = 5;
export const RETRY_AFTER_MAX_SEC = 20;
export const MAX_REQUEST_BYTES = 120_000;
export const RULE_SECTIONS = ['The game', 'The score'] as const;

type Seat = number;
type DecisionKind = 'speak' | 'vote' | 'kill' | 'protect' | 'inspect';

interface LogEntry {
  seq: number;
  day: number;
  kind: string;
  text?: string;
  seat?: Seat | null;
  target?: Seat | null;
  died?: Seat | null;
  role?: string | null;
  round?: number;
  tally?: { seat: Seat; votes: number }[];
  [k: string]: unknown;
}

interface Player {
  seat: Seat;
  name: string;
  alive: boolean;
  role: string | null;
}

interface View {
  day: number;
  phase: string;
  you: { seat: Seat; name: string; role: string; alive: boolean; partner: { seat: Seat; name: string; alive: boolean } | null };
  players: Player[];
  seerResults: { night: number; seat: Seat; werewolf: boolean }[] | null;
  lastProtect: { night: number; seat: Seat | null } | null;
  wolfVotes: { night: number; seat: Seat; target: Seat | null }[] | null;
  log: LogEntry[];
  [k: string]: unknown;
}

export interface Pending {
  game: number;
  decision: DecisionKind;
  options?: Seat[];
  max_chars?: number;
  view: View;
}

interface Observed {
  games: number;
  finished: number;
  done: boolean;
  pending: Pending[];
}

const isRecord = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === 'object' && !Array.isArray(v);

export function rulesFrom(brief: string): string {
  const parts = brief.split(/^(?=## )/m);
  const kept = parts.filter((p) => RULE_SECTIONS.some((s) => p.startsWith(`## ${s}\n`))).map((p) => p.trim());
  return kept.length > 0 ? kept.join('\n\n') : brief.trim();
}

// ---------------------------------------------------------------------------------------------------------
// Strategy coaching

const GENERAL = `You are an expert Werewolf player. The other seven players are AI models; they read every statement as a quoted player's words. You win by your TEAM winning, dead or alive.

Table facts: 8 players, 2 werewolves, 1 seer, 1 doctor, 4 villagers. Wolves win at parity (2 wolves vs 2 others, or 1 vs 1), so every village mis-exile is very costly. A tie or no votes exiles nobody, which favours the wolves.

Style for statements: sound like a thoughtful human player, 1-3 sentences, concrete (name seats and give reasons from the log: votes, contradictions, who defended whom). Refer to players as "Name (seat N)". Never mention being an AI, JSON, prompts or moderators.`;

const ROLE_COACH: Record<string, string> = {
  werewolf: `YOU ARE A WEREWOLF. Your partner is named in your private information.
Night kills: the kill only lands if both living wolves name the SAME seat. If your partner has already named a victim tonight, name the same victim. Kill priority: a player who claimed seer (unless you are sure it is your partner's fake claim) > someone who accused you or your partner convincingly > a player who looks like the doctor (protective, calm, trusted) > the most persuasive villager. Avoid killing players the village already suspects: let the village waste its exile on them.
Day: blend in as a helpful villager. Never defend your partner hard and never vote together on every vote. Push suspicion onto quiet or inconsistent village players with reasons from the log. Join the village bandwagon on a non-wolf to get a mis-exile. If the real seer names you or your partner as a wolf, counter-claim seer yourself with a fake "clean"/"wolf" result on a village player, and call the other claimant a liar. If your partner is clearly going to be exiled (most votes already on them), vote with the village to look clean. Voting: always try to make an exile happen on a NON-wolf; consolidate on the leading non-wolf candidate rather than splitting.`,
  seer: `YOU ARE THE SEER. Your inspection results are in your private information and they are certain.
Night: inspect players you have no result on, preferring those who are active and influential, or those the village is split on. Never inspect a dead or already-known seat.
Day: if you found a werewolf, claim seer at once and name that wolf with your result, then lead the vote onto them. If you only have "not a werewolf" results, you may hold them quietly on day 1 (to survive), but reveal everything once you find a wolf, once someone else falsely claims seer (they are a wolf: say so), or when your death or a wrong exile looks likely. When you claim, list ALL your results by night. Always vote for a known living werewolf.`,
  doctor: `YOU ARE THE DOCTOR. Your last protection is in your private information; you cannot protect yourself or the same player two nights running.
Night: protect a player who claimed seer and seems genuine (if there is a counter-claim, protect the one more consistent with the dead players' roles). With no seer claim, protect the most trusted, most active village voice who the wolves would want dead.
Day: do not reveal you are the doctor unless it saves the village (e.g., you are about to be exiled). Help the village find wolves: study votes (wolves avoid voting their partner), defences, and seer claims. Back a credible seer's accusation and consolidate votes so an exile happens.`,
  villager: `YOU ARE A VILLAGER with no special information.
Day: find the wolves from evidence. Wolves rarely vote for each other, often defend each other, push vague accusations, and jump on bandwagons late. A single seer claim is usually genuine; if two people claim seer, one is a wolf, and the one whose results fit the deaths and exiled roles is more likely real. Back a credible seer's named wolf. Consolidate the vote: an exile with a good reason beats a split that exiles nobody, but avoid exiling a player the evidence says is clean. Do not claim a power role.`,
};

function systemFor(rules: string, role: string): string {
  return [GENERAL, '', ROLE_COACH[role] ?? ROLE_COACH.villager, '', 'THE RULES', rules].join('\n');
}

// ---------------------------------------------------------------------------------------------------------
// A readable digest of the view

const nameOf = (v: View, s: Seat | null | undefined): string => {
  if (s === null || s === undefined) return 'nobody';
  const p = v.players.find((x) => x.seat === s);
  return p ? `${p.name} (seat ${s})` : `seat ${s}`;
};

function logLine(v: View, e: LogEntry): string {
  switch (e.kind) {
    case 'night':
      return `-- Night ${e.day} --`;
    case 'dawn':
      return e.died === null || e.died === undefined ? `Dawn of day ${e.day}: no one died.` : `Dawn of day ${e.day}: ${nameOf(v, e.died)} was killed in the night; they were a ${e.role}.`;
    case 'debate':
      return `-- Day ${e.day} debate --`;
    case 'speech':
      return `${nameOf(v, e.seat)} [round ${e.round}]: "${e.text}"`;
    case 'pass':
      return `${nameOf(v, e.seat)} [round ${e.round}] passed.`;
    case 'voting':
      return `-- Day ${e.day} vote --`;
    case 'vote':
      return `${nameOf(v, e.seat)} votes ${e.target === null ? 'to abstain' : `to exile ${nameOf(v, e.target)}`}.`;
    case 'exile': {
      const tally = (e.tally ?? []).map((t) => `${nameOf(v, t.seat)}: ${t.votes}`).join(', ') || 'no votes';
      return e.seat === null || e.seat === undefined ? `Nobody was exiled (${tally}).` : `${nameOf(v, e.seat)} was exiled (${tally}); they were a ${e.role}.`;
    }
    case 'end':
      return `Game over.`;
    default:
      return JSON.stringify(e);
  }
}

/** Today's vote so far, as a tally, when the day's vote has begun. */
function todaysVotes(v: View): Map<Seat, number> {
  const tally = new Map<Seat, number>();
  let inVote = false;
  for (const e of v.log) {
    if (e.kind === 'voting') {
      inVote = e.day === v.day;
      tally.clear();
    } else if (e.kind === 'exile') inVote = false;
    else if (e.kind === 'vote' && inVote && typeof e.target === 'number') tally.set(e.target, (tally.get(e.target) ?? 0) + 1);
  }
  return inVote ? tally : new Map();
}

/** Seats whose statements claim seer (a mention of "seer" with "I am"/"I'm"). */
function seerClaimants(v: View): Seat[] {
  const out: Seat[] = [];
  for (const e of v.log) {
    if (e.kind !== 'speech' || typeof e.seat !== 'number' || typeof e.text !== 'string') continue;
    if (/\b(i am|i'm|im)\s+(the\s+)?seer\b/i.test(e.text) || /\bas (the )?seer\b/i.test(e.text)) if (!out.includes(e.seat)) out.push(e.seat);
  }
  return out;
}

function digest(v: View): string {
  const lines: string[] = [];
  lines.push(`You are ${nameOf(v, v.you.seat)}, the ${v.you.role}, ${v.you.alive ? 'alive' : 'dead'}. It is ${v.phase === 'night' ? `night ${v.day}` : `day ${v.day} (${v.phase})`}.`);
  if (v.you.partner) lines.push(`Your werewolf partner: ${nameOf(v, v.you.partner.seat)}, ${v.you.partner.alive ? 'alive' : 'dead'}.`);
  lines.push('', 'PLAYERS');
  for (const p of v.players) lines.push(`- ${nameOf(v, p.seat)}: ${p.alive ? 'alive' : `dead, was a ${p.role}`}${p.seat === v.you.seat ? ' (you)' : ''}${v.you.partner && p.seat === v.you.partner.seat ? ' (your partner)' : ''}`);
  const alive = v.players.filter((p) => p.alive);
  lines.push(`Alive: ${alive.length}.`);
  lines.push('', 'YOUR PRIVATE INFORMATION');
  if (v.seerResults) {
    if (v.seerResults.length === 0) lines.push('- No inspection results yet.');
    for (const r of v.seerResults) lines.push(`- Night ${r.night}: you inspected ${nameOf(v, r.seat)}: ${r.werewolf ? 'A WEREWOLF' : 'not a werewolf'}.`);
  }
  if (v.lastProtect) lines.push(`- Night ${v.lastProtect.night}: you protected ${nameOf(v, v.lastProtect.seat)} (you cannot protect them tonight if that was last night).`);
  if (v.wolfVotes) {
    for (const w of v.wolfVotes) lines.push(`- Night ${w.night}: ${nameOf(v, w.seat)} named ${nameOf(v, w.target)} as victim.`);
  }
  if (!v.seerResults && !v.lastProtect && !v.wolfVotes) lines.push('- None.');
  const claims = seerClaimants(v);
  if (claims.length > 0) lines.push('', `Seer claims so far (by statement text): ${claims.map((s) => nameOf(v, s)).join(', ')}.`);
  lines.push('', 'GAME LOG');
  for (const e of v.log) lines.push(logLine(v, e));
  const tally = todaysVotes(v);
  if (tally.size > 0) lines.push('', `Today's vote so far: ${[...tally.entries()].sort((a, b) => b[1] - a[1]).map(([s, n]) => `${nameOf(v, s)} ${n}`).join(', ')}.`);
  return lines.join('\n');
}

function decisionText(p: Pending): string {
  const v = p.view;
  const opts = (p.options ?? []).map((s) => nameOf(v, s)).join(', ');
  switch (p.decision) {
    case 'speak':
      return `It is your turn to speak in the debate. Make one statement of at most ${Math.min(p.max_chars ?? 500, 450)} characters, or pass (passing is usually weak; speak).\nReply with one JSON object and nothing after it: {"say": "<your statement>"}, or {"say": null} to pass.`;
    case 'vote':
      return `It is your turn to vote. Vote to exile one of: ${opts}; or abstain.\nReply with one JSON object and nothing after it: {"target": <seat number>}, or {"target": null}.`;
    case 'kill':
      return `It is night: name the victim, one of: ${opts}; or abstain.\nReply with one JSON object and nothing after it: {"target": <seat number>}, or {"target": null}.`;
    case 'protect':
      return `It is night: choose the seat to protect, one of: ${opts}; or nobody.\nReply with one JSON object and nothing after it: {"target": <seat number>}, or {"target": null}.`;
    case 'inspect':
      return `It is night: choose the seat to inspect, one of: ${opts}; or nobody.\nReply with one JSON object and nothing after it: {"target": <seat number>}, or {"target": null}.`;
  }
}

function userText(p: Pending, maxLog?: number): string {
  let v = p.view;
  if (maxLog !== undefined && v.log.length > maxLog) v = { ...v, log: v.log.slice(v.log.length - maxLog) };
  return [digest(v), '', `YOUR DECISION (${p.decision})`, decisionText(p)].join('\n');
}

const requestBytes = (r: Record<string, unknown>): number => Buffer.byteLength(JSON.stringify(r), 'utf8');

function bodyOf(system: string, user: string, thinking: boolean): Record<string, unknown> {
  return {
    model: MODEL,
    max_tokens: MAX_TOKENS,
    ...(thinking ? { thinking: { type: 'adaptive' }, output_config: { effort: EFFORT } } : {}),
    system: [{ type: 'text', text: system, cache_control: { type: 'ephemeral' } }],
    messages: [{ role: 'user', content: user }],
  };
}

export function requestFor(rules: string, p: Pending, thinking: boolean): Record<string, unknown> {
  const system = systemFor(rules, p.view.you.role);
  let req = bodyOf(system, userText(p), thinking);
  let keep = p.view.log.length;
  while (requestBytes(req) > MAX_REQUEST_BYTES && keep > 10) {
    keep = Math.floor(keep * 0.8);
    req = bodyOf(system, userText(p, keep), thinking);
  }
  return req;
}

// ---------------------------------------------------------------------------------------------------------
// Reply parsing

function jsonObjects(text: string): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    let depth = 0;
    let inString = false;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inString) {
        if (ch === '\\') i += 1;
        else if (ch === '"') inString = false;
      } else if (ch === '"') inString = true;
      else if (ch === '{') depth += 1;
      else if (ch === '}') {
        depth -= 1;
        if (depth === 0) {
          try {
            const v: unknown = JSON.parse(text.slice(start, i + 1));
            if (isRecord(v)) out.push(v);
          } catch {
            // not JSON
          }
          break;
        }
      }
    }
  }
  return out;
}

export function foldStatement(text: string): string {
  return text
    .replace(new RegExp('[\t\n\v\f\r' + String.fromCharCode(0x85, 0x2028, 0x2029) + ']+', 'g'), ' ')
    .replace(/\p{Cc}/gu, '')
    .trim();
}

/** The model's answer: a statement string, a pass (null), a seat, an abstention (null), or undefined when unreadable. */
function parseAnswer(text: string, p: Pending): { say?: string | null; target?: Seat | null } | undefined {
  const key = p.decision === 'speak' ? 'say' : 'target';
  const found = jsonObjects(text).filter((o) => key in o);
  if (found.length === 0) return undefined;
  const value = found[found.length - 1]![key];
  if (p.decision === 'speak') {
    if (value === null) return { say: null };
    if (typeof value !== 'string') return undefined;
    const folded = foldStatement(value);
    if (folded === '') return { say: null };
    const max = p.max_chars ?? 500;
    return { say: [...folded].length > max ? [...folded].slice(0, max).join('').trim() : folded };
  }
  const seat = typeof value === 'string' && /^\s*\d+\s*$/.test(value) ? Number(value) : value;
  if (seat === null) return { target: null };
  if (typeof seat === 'number' && (p.options ?? []).includes(seat)) return { target: seat };
  return undefined;
}

function replyText(body: unknown): string {
  const content = isRecord(body) ? body.content : undefined;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

// ---------------------------------------------------------------------------------------------------------
// Deterministic picks and the hard rules

let TRIAL_SEED = '';
function pick<T>(items: readonly T[], game: number, salt: string): T | undefined {
  if (items.length === 0) return undefined;
  const h = createHash('sha256').update(`${TRIAL_SEED}|${game}|${salt}`).digest();
  return items[h.readUInt32BE(0) % items.length];
}

const knownWolves = (v: View): Seat[] => (v.seerResults ?? []).filter((r) => r.werewolf).map((r) => r.seat);
const knownClean = (v: View): Seat[] => (v.seerResults ?? []).filter((r) => !r.werewolf).map((r) => r.seat);
const aliveSeats = (v: View): Seat[] => v.players.filter((p) => p.alive).map((p) => p.seat);

/** Who has voted for whom over the whole game (village votes as suspicion evidence). */
function accusersOf(v: View, seat: Seat): Seat[] {
  return v.log.filter((e) => e.kind === 'vote' && e.target === seat && typeof e.seat === 'number').map((e) => e.seat as Seat);
}

/** The scripted target for a decision: used when the model gave nothing usable. */
function scriptedTarget(p: Pending): Seat | null {
  const v = p.view;
  const opts = p.options ?? [];
  const salt = `${p.decision}:${v.log.length}`;
  const partner = v.you.partner?.seat;
  const tally = todaysVotes(v);
  const leader = [...tally.entries()].filter(([s]) => opts.includes(s)).sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  const claims = seerClaimants(v);
  switch (p.decision) {
    case 'kill': {
      const claimed = claims.filter((s) => opts.includes(s) && s !== partner);
      if (claimed.length > 0) return claimed[0]!;
      const accusers = opts.filter((s) => accusersOf(v, v.you.seat).includes(s) || (partner !== undefined && accusersOf(v, partner).includes(s)));
      return pick(accusers.length > 0 ? accusers : opts, p.game, salt) ?? null;
    }
    case 'protect': {
      const claimed = claims.filter((s) => opts.includes(s));
      if (claimed.length > 0) return claimed[0]!;
      return pick(opts, p.game, salt) ?? null;
    }
    case 'inspect': {
      const known = new Set((v.seerResults ?? []).map((r) => r.seat));
      const fresh = opts.filter((s) => !known.has(s));
      return pick(fresh.length > 0 ? fresh : opts, p.game, salt) ?? null;
    }
    case 'vote': {
      if (v.you.role === 'werewolf') {
        const nonWolf = leader.filter(([s]) => s !== partner);
        if (nonWolf.length > 0) return nonWolf[0]![0];
        return pick(opts.filter((s) => s !== partner), p.game, salt) ?? null;
      }
      const wolf = knownWolves(v).find((s) => opts.includes(s));
      if (wolf !== undefined) return wolf;
      const clean = new Set(knownClean(v));
      const lead = leader.filter(([s]) => !clean.has(s));
      if (lead.length > 0) return lead[0]![0];
      return pick(opts.filter((s) => !clean.has(s)), p.game, salt) ?? null;
    }
    default:
      return null;
  }
}

function scriptedSay(p: Pending): string | null {
  const v = p.view;
  if (v.you.role === 'seer' && (v.seerResults ?? []).length > 0) {
    const wolves = knownWolves(v).filter((s) => aliveSeats(v).includes(s));
    const res = (v.seerResults ?? []).map((r) => `night ${r.night}: ${nameOf(v, r.seat)} is ${r.werewolf ? 'a WEREWOLF' : 'not a werewolf'}`).join('; ');
    if (wolves.length > 0) return `I am the seer. My results: ${res}. Vote out ${nameOf(v, wolves[0])} today.`;
  }
  return null;
}

/** The hard rules applied to a target the model (or script) chose. */
function enforceTarget(p: Pending, chosen: Seat | null): Seat | null {
  const v = p.view;
  const opts = p.options ?? [];
  const partner = v.you.partner;
  if (p.decision === 'kill') {
    const mine = (v.wolfVotes ?? []).filter((w) => w.night === v.day && w.seat !== v.you.seat && w.target !== null && opts.includes(w.target));
    if (mine.length > 0) return mine[mine.length - 1]!.target; // agree with the partner's pick
    if (chosen === null || !opts.includes(chosen)) return scriptedTarget(p);
    return chosen;
  }
  if (p.decision === 'inspect') {
    const known = new Set((v.seerResults ?? []).map((r) => r.seat));
    if (chosen === null || known.has(chosen) || !opts.includes(chosen)) return scriptedTarget(p);
    return chosen;
  }
  if (p.decision === 'protect') {
    if (chosen === null || !opts.includes(chosen)) return scriptedTarget(p);
    return chosen;
  }
  if (p.decision === 'vote') {
    if (v.you.role === 'werewolf' && partner) {
      if (chosen === partner.seat) {
        // Bussing the partner is allowed only when they are already lost (a majority of the living on them).
        const onPartner = todaysVotes(v).get(partner.seat) ?? 0;
        if (onPartner * 2 < aliveSeats(v).length - 1) return scriptedTarget(p);
      }
      return chosen === null ? scriptedTarget(p) : chosen;
    }
    if (v.you.role === 'seer') {
      const wolf = knownWolves(v).find((s) => opts.includes(s));
      if (wolf !== undefined) return wolf;
    }
    if (chosen !== null && knownClean(v).includes(chosen)) return scriptedTarget(p);
    return chosen === null ? scriptedTarget(p) : chosen;
  }
  return chosen;
}

// ---------------------------------------------------------------------------------------------------------
// The run

interface Reply {
  status: number;
  body: unknown;
  retryAfter: string | null;
}

async function readReply(res: Response): Promise<Reply> {
  const text = await res.text();
  let body: unknown = null;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  return { status: res.status, body, retryAfter: res.headers.get('retry-after') };
}

const SPENT = new Set(['budget_exhausted', 'request_cap']);
const errorType = (body: unknown): string => {
  const e = isRecord(body) ? body.error : undefined;
  return isRecord(e) && typeof e.type === 'string' ? e.type : '';
};
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));
function retryWaitMs(retryAfter: string | null): number {
  const v = (retryAfter ?? '').trim();
  if (!/^\d+$/.test(v)) return RETRY_WAIT_SEC * 1000;
  return Math.min(Number(v), RETRY_AFTER_MAX_SEC) * 1000;
}

export async function run(): Promise<void> {
  const env = process.env;
  const envUrl = (env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  const llmUrl = (env.ANTHROPIC_BASE_URL ?? '').replace(/\/+$/, '');
  const apiKey = env.ANTHROPIC_API_KEY ?? '';
  TRIAL_SEED = env.MAKEFUN_TRIAL_SEED ?? '';
  if (envUrl === '') throw new Error('MAKEFUN_ENV_URL is needed');
  const brief = env.MAKEFUN_PROMPT_FILE ? readFileSync(env.MAKEFUN_PROMPT_FILE, 'utf8') : '';
  const rules = rulesFrom(brief);
  let decisions = 0;
  let modelAnswers = 0;
  let calls = 0;

  const envPost = async (path: string, body: unknown): Promise<Reply> =>
    readReply(await fetch(`${envUrl}${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }));
  const messages = async (request: Record<string, unknown>): Promise<Reply> => {
    calls += 1;
    return readReply(
      await fetch(`${llmUrl}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'anthropic-version': ANTHROPIC_VERSION, 'x-api-key': apiKey },
        body: JSON.stringify(request),
      }),
    );
  };

  const ask = async (p: Pending): Promise<Reply | null> => {
    if (llmUrl === '') return null;
    let thinking = true;
    let reply = await messages(requestFor(rules, p, thinking));
    if (reply.status === 400 && errorType(reply.body) === 'invalid_request_error') {
      thinking = false;
      reply = await messages(requestFor(rules, p, thinking));
    }
    if (SPENT.has(errorType(reply.body)) || reply.status === 401) return null;
    if (reply.status === 429 || reply.status >= 500) {
      await sleep(retryWaitMs(reply.retryAfter));
      reply = await messages(requestFor(rules, p, thinking));
    }
    return reply.status === 200 ? reply : null;
  };

  const decide = async (p: Pending): Promise<void> => {
    decisions += 1;
    let answer: ReturnType<typeof parseAnswer>;
    try {
      const reply = await ask(p);
      if (reply !== null) answer = parseAnswer(replyText(reply.body), p);
    } catch {
      answer = undefined;
    }
    if (answer !== undefined) modelAnswers += 1;
    let act: Record<string, unknown>;
    if (p.decision === 'speak') {
      const say = answer?.say ?? (answer === undefined ? scriptedSay(p) : null);
      act = say ? { game: p.game, decision: 'speak', text: say } : { game: p.game, decision: 'speak', pass: true };
    } else {
      const chosen = answer === undefined ? scriptedTarget(p) : (answer.target ?? null);
      act = { game: p.game, decision: p.decision, target: enforceTarget(p, chosen) };
    }
    const posted = await envPost('/act', act);
    if (posted.status === 400) {
      const fb = p.decision === 'speak' ? { game: p.game, decision: 'speak', pass: true } : { game: p.game, decision: p.decision, target: null };
      await envPost('/act', fb);
    }
  };

  const keyOf = (p: Pending): string => `${p.game}:${p.decision}:${p.view.log.length}`;
  const inFlight = new Map<string, Promise<void>>();
  const started = new Set<string>();
  const errors: string[] = [];
  for (;;) {
    const r = await envPost('/observe', {});
    if (r.status !== 200 || !isRecord(r.body)) throw new Error(`POST /observe answered ${r.status}`);
    const obs = r.body as unknown as Observed;
    if (obs.done) break;
    for (const p of obs.pending) {
      const key = keyOf(p);
      if (started.has(key) || inFlight.size >= CONCURRENCY) continue;
      started.add(key);
      const task = decide(p)
        .catch((err: unknown) => {
          errors.push(`game ${p.game}: ${(err as Error).message}`);
        })
        .finally(() => inFlight.delete(key));
      inFlight.set(key, task);
    }
    const waiting = obs.pending.some((p) => !started.has(keyOf(p)));
    if (inFlight.size > 0 && (inFlight.size >= CONCURRENCY || !waiting)) await Promise.race(inFlight.values());
    else if (inFlight.size === 0 && obs.pending.length > 0 && !waiting) await sleep(250);
    if (errors.length > 0) break;
  }
  await Promise.all(inFlight.values());
  if (errors.length > 0) throw new Error(`the env failed: ${errors.join('; ')}`);
  process.stdout.write(`agent: ${decisions} decisions, ${modelAnswers} answered by the model, ${calls} model calls\n`);
}

const isMain = process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  try {
    await run();
    process.exitCode = 0;
  } catch (err) {
    process.stderr.write(`agent: ${(err as Error).stack ?? String(err)}\n`);
    process.exitCode = 1;
  }
}
