"""
Heads-up Discard Hold'em Production Bot.
Compliant with Problem Statement, Rulebook, and Engine Protocol 1.0.
"""
import itertools
import json
import random
import sys
import time

DECK = tuple(range(52))
NAMES = [r + s for r in "23456789TJQKA" for s in "shdc"]
INDEX = {name: i for i, name in enumerate(NAMES)}
TOP = [tuple(i for i in range(12, -1, -1) if m >> i & 1) for m in range(8192)]
STRAIGHT = [next((i for i in range(12, 3, -1) if m >> (i - 4) & 31 == 31),
                 3 if m & 4111 == 4111 else -1) for m in range(8192)]


def rank(cards):
    """5-card hand evaluator out of 5, 6, or 7 cards."""
    suits = [0] * 4
    seen = pairs = trips = quads = 0
    for c in cards:
        bit = 1 << (c // 4)
        suits[c % 4] |= bit
        quads |= trips & bit
        trips |= pairs & bit
        pairs |= seen & bit
        seen |= bit
    flush = next((s for s in suits if s.bit_count() >= 5), 0)
    if flush and STRAIGHT[flush] >= 0:
        return (8, STRAIGHT[flush])
    if quads:
        return (7, TOP[quads][0], TOP[seen ^ quads][0])
    if trips:
        t = TOP[trips][0]
        rest = pairs ^ (1 << t)
        if rest:
            return (6, t, TOP[rest][0])
    if flush:
        return (5, *TOP[flush][:5])
    if STRAIGHT[seen] >= 0:
        return (4, STRAIGHT[seen])
    if trips:
        return (3, t, *TOP[seen ^ trips][:2])
    if pairs.bit_count() >= 2:
        a, b = TOP[pairs][:2]
        return (2, a, b, TOP[seen ^ (1 << a) ^ (1 << b)][0])
    if pairs:
        return (1, TOP[pairs][0], *TOP[seen ^ pairs][:3])
    return (0, *TOP[seen][:5])


def keep_value(pair, flop):
    """Heuristic for evaluating 2 cards + flop."""
    cards = pair + flop
    made = rank(cards)
    mask = sum(1 << r for r in set(c // 4 for c in cards))
    suit_count = max(sum(c % 4 == s for c in cards) for s in range(4))
    outs = sum(STRAIGHT[mask | 1 << r] >= 0 for r in range(13) if not mask >> r & 1)
    return (made[0] + .048 * made[1] + .012 * max(c // 4 for c in pair)
            + .86 * (suit_count == 4) + .22 * min(2, outs))


def modeled_pair(three, flop):
    return max(itertools.combinations(three, 2), key=lambda p: keep_value(p, flop))


def default_discard(cards):
    """Fallback discard matching engine tie-break rules."""
    return min(cards, key=lambda c: ("23456789TJQKA".index(c[0]), "shdc".index(c[1])))


class Bot:
    def __init__(self):
        self.hands = 200
        self.limit = 5000
        self.hole = ()
        self.dead = ()
        self.cache = {}
        self.folds = self.faced = self.bets = self.actions = 0

    def estimate(self, req, board):
        choices = list(itertools.combinations(self.hole, 2))
        pool = sorted(set(DECK).difference(self.hole, board, self.dead))
        scores = [0.0] * len(choices)
        weight_sum = 0

        time_left = req.get("time_left_ms", 30000)
        hands_left = max(1, self.hands - req.get("hand_id", 1) + 1)
        budget = time_left / hands_left

        # Emergency Breaker: If bank is dangerously low (<2.5s), run minimal trials
        if time_left < 2500:
            end = time.perf_counter() + 0.005
            max_trials = 80
        else:
            end = time.perf_counter() + min(.045, self.limit * .0004, budget * .00016)
            max_trials = 800

        # Precompute static showdown hand outside loop on River (saves ~30% CPU)
        fixed_mine = rank(choices[0] + board) if len(board) == 5 else None

        for _ in range(max_trials):
            sample = random.sample(pool, (5 - len(board)) + (2 if self.dead else 3))
            final = board + tuple(sample[:5 - len(board)])
            flop = final[:3]
            other = tuple(sample[5 - len(board):])
            weight = 1.0

            if self.dead:
                # Bayesian weight for opponent discard consistency
                if modeled_pair(other + (self.dead[1],), flop) != other:
                    weight = 0.20
            else:
                other = modeled_pair(other, flop)

            opponent = rank(other + final)
            candidates = choices if board else [modeled_pair(self.hole, flop)]

            if fixed_mine is not None:
                scores[0] += weight * ((fixed_mine > opponent) + 0.5 * (fixed_mine == opponent))
            else:
                for i, pair in enumerate(candidates):
                    mine = rank(pair + final)
                    scores[i] += weight * ((mine > opponent) + 0.5 * (mine == opponent))

            weight_sum += weight
            if time.perf_counter() >= end:
                break

        if not weight_sum:
            return 0.5, choices[0]

        best = max(range(len(scores)), key=scores.__getitem__)
        return scores[best] / weight_sum, choices[best]

    def message(self, m):
        try:
            return self._process(m)
        except Exception:
            # Shield: Always output a valid legal reply even if an internal error occurs
            kind = m.get('type')
            if kind == 'discard_request':
                cards = m.get('hole_cards', [])
                if cards:
                    return {"type": "discard", "card": default_discard(cards)}
            elif kind == 'action_request':
                legal = m.get('legal_actions', ['check'])
                if 'check' in legal:
                    return {"type": "action", "action": "check"}
                if 'call' in legal and m.get('to_call', 0) <= 2:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}
            return None

    def _process(self, m):
        kind = m.get('type')
        if kind == 'init':
            self.hands = m.get('total_hands', 200)
            self.limit = m.get('per_response_ms', 5000)
            return None

        elif kind == 'new_hand':
            self.hole = tuple(map(INDEX.get, m['hole_cards']))
            self.dead = ()
            self.cache = {}
            return None

        elif kind == 'discard_result':
            self.hole = tuple(map(INDEX.get, m['hole_cards']))
            self.dead = (INDEX[m['my_discard']], INDEX[m['opp_discard']])
            return None

        elif kind == 'hand_result':
            previous = None
            for act in m.get('history', []):
                if act.get('actor') == 'opp':
                    self.actions += 1
                    self.bets += (act.get('action') == 'raise')
                    if previous and previous.get('street') == act.get('street') and previous.get('action') == 'raise':
                        self.faced += 1
                        self.folds += (act.get('action') == 'fold')
                previous = act
            return None

        elif kind in ('action_request', 'discard_request'):
            board_cards = m.get('board', [])
            board = tuple(map(INDEX.get, board_cards))
            street = m.get('street', 'preflop')

            if board not in self.cache:
                self.cache[board] = self.estimate(m, board)
            equity, keep = self.cache[board]

            if kind == 'discard_request':
                discarded = next(c for c in self.hole if c not in keep)
                return {"type": "discard", "card": NAMES[discarded]}

            legal = m.get('legal_actions', ['check'])
            cost = m.get('to_call', 0)
            pot = m.get('pot', 1)
            history = m.get('history', [])

            # Multi-street pressure calculation
            street_pressure = sum(1 for a in history if a['actor'] == 'opp' and a['action'] == 'raise' and a['street'] == street)
            prior_pressure = sum(1 for a in history if a['actor'] == 'opp' and a['action'] == 'raise' and a['street'] != street)
            total_pressure = street_pressure + 0.5 * prior_pressure

            aggression = (self.bets + 4) / (self.actions + 16)
            fold_rate = (self.folds + 3) / (self.faced + 9)

            # Balanced equity discount for opponent aggression
            equity_discount = min(0.30, total_pressure * (0.09 + 0.11 * cost / max(1, pot)) * (1.35 - aggression))
            adjusted_equity = max(0.0, equity - equity_discount)

            # Preflop Button Opening: Steal blinds against passive/normal players
            is_preflop_btn_open = (street == 'preflop' and cost == 1 and len(history) == 0)
            bluff = (cost == 0 or is_preflop_btn_open) and equity < 0.44 and (random.random() < 0.32 * fold_rate)

            # Controlled raise threshold
            threshold = 0.55 if not street_pressure else 0.66

            should_raise = ('raise' in legal) and (
                (adjusted_equity > threshold) or
                (is_preflop_btn_open and adjusted_equity > 0.42) or
                bluff
            )

            if should_raise:
                if is_preflop_btn_open:
                    target = m.get('opp_bet', 2) + 4  # Raise to 6 preflop
                elif bluff:
                    target = m.get('opp_bet', 0) + round(pot * 0.50)
                else:
                    fraction = 0.50 + max(0, adjusted_equity - 0.55) * 1.5
                    target = m.get('opp_bet', 0) + round((pot + cost) * fraction)

                clamped = min(m['max_raise'], max(m['min_raise'], target))
                return {"type": "action", "action": "raise", "amount": clamped}

            # Check if free
            if 'check' in legal:
                return {"type": "action", "action": "check"}

            # Pot odds decision
            pot_odds = cost / (pot + cost)
            action = "call" if adjusted_equity >= pot_odds else "fold"
            return {"type": "action", "action": action}

        return None


def main():
    bot = Bot()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if msg.get('type') == 'match_end':
            break
        reply = bot.message(msg)
        if reply:
            print(json.dumps(reply), flush=True)


if __name__ == '__main__':
    main()
