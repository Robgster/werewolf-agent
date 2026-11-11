# Werewolf agent

An entry for make.fun's [Build the best Werewolf agent](https://make.fun/bounty/werewolf-agent-01) bounty.

`main.ts` (Node 22, standard library only) plays one seat in each of 30 eight-player Werewolf games against house players.

- Every decision goes to `claude-opus-5-5` once, with role-specific strategy coaching and a readable digest of that game's view: players, private knowledge, the public log, today's vote tally, and seer claims.
- Hard rules then correct the model where it must not slip:
  - The second wolf copies its partner's night pick, so the kill always lands.
  - A wolf never votes out its partner unless the partner is already lost.
  - The seer never re-inspects a known seat and always votes a werewolf it found.
  - Village roles never vote a seat the seer knows is clean.
- If no model reply is usable (no key pushed, budget spent, or an unreadable reply), a scripted player answers instead of passing.
- The agent is deterministic given the model's replies: picks come from a hash of `MAKEFUN_TRIAL_SEED`, the game number and the log position, never from the clock.

License: MIT.
